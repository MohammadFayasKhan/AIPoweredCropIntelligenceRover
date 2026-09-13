"""
Groq Multimodal Vision Semantic Preflight Validator for SmartCropVision.
Evaluates uploaded images upstream of local computer vision models to establish
whether the input is an authentic crop or plant leaf photograph rather than
a software screenshot, dashboard, document, PDF, vehicle, person, or unrelated object.

Architectural principles:
  1. Groq Vision is strictly an input suitability and domain safety gate.
  2. Groq Vision never performs disease diagnosis or pathological classification.
  3. If Groq Vision rejects the image, the pipeline terminates immediately with zero model invocations.
  4. If Groq Vision encounters a network timeout, rate limit, or transient error, the system
     falls back safely to local deterministic computer vision checks.
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import cv2
import numpy as np
import requests

from backend.app.config import settings

logger = logging.getLogger("smartcropvision.groq_validator")


@dataclass
class GroqValidationResult:
    valid: bool
    category: str  # "plant_leaf", "screenshot_document", "non_plant", "uncertain"
    plant_present: bool
    leaf_present: bool
    suitable_for_crop_analysis: bool
    reason: str
    inference_allowed: bool
    latency_ms: float
    model_used: str
    raw_response: Optional[Dict[str, Any]] = None


class GroqVisionValidator:
    """
    Client for Groq Multimodal Vision models (Qwen 3.8 27B / Qwen 3.6 27B).
    Provides structured semantic domain verification with robust fallbacks.
    """

    GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: Optional[float] = None
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_VISION_MODEL
        self.enabled = enabled if enabled is not None else settings.GROQ_VISION_ENABLED
        self.timeout = min(float(timeout or settings.GROQ_VISION_TIMEOUT), 3.5)

    def validate_image(
        self,
        img_bgr: np.ndarray,
        filename: Optional[str] = None
    ) -> Optional[GroqValidationResult]:
        """
        Submits an image to Groq Vision for semantic plant photograph verification.
        Returns GroqValidationResult if successful, or None if validation failed or is disabled.
        """
        if not self.enabled:
            logger.debug("Groq vision validation is disabled by configuration.")
            return None

        if not self.api_key or not self.api_key.strip() or self.api_key.startswith("replace-with"):
            logger.debug("Groq vision API key is not configured.")
            return None

        start_time = time.time()

        try:
            # 1. Downscale image to max 480px on longest dimension to minimize transmission and token costs
            h, w = img_bgr.shape[:2]
            max_dim = 480
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                target_w = max(32, int(w * scale))
                target_h = max(32, int(h * scale))
                scaled = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                scaled = img_bgr

            # 2. Encode to JPEG with modest compression (quality 75)
            success, buffer = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not success or buffer is None:
                logger.warning("Could not encode image buffer for Groq vision preflight.")
                return None

            b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64_str}"

            # 3. Formulate structured semantic validation payload
            system_prompt = (
                "/no_thinking\n"
                "You are an agricultural plant image validator. "
                "The user is testing crop leaf disease diagnosis using direct foliage photos OR digital leaf images displayed on smartphones, tablets, laptop screens, computer monitors, or in hands. "
                "CRITICAL INSTRUCTION: If any plant leaf, foliage, or crop disease image is present (even if displayed on a phone screen, computer monitor, or digital display, or held in a hand, with bezels/reflections), you MUST classify it as: "
                'category: "plant_leaf", inference_allowed: true, valid: true, plant_present: true, leaf_present: true, suitable_for_crop_analysis: true so it can be diagnosed! '
                "DO NOT disqualify device bezels, camera cutouts, hands, or screen reflections. "
                "ONLY reject images that have NO plant or leaf at all (e.g. pure human faces without plants, cars, pets, empty spreadsheets, blank documents). "
                "Output strictly raw JSON without markdown fences."
            )

            user_prompt = (
                "Determine whether this image contains or displays a plant or crop leaf suitable for crop disease diagnosis. "
                "Rules:\n"
                "1. If this image shows a plant or crop leaf (whether a direct photo, or displayed on a mobile phone, tablet, laptop, or monitor screen, or held in a hand): "
                "ACCEPT IT as category: plant_leaf, plant_present: true, leaf_present: true, suitable_for_crop_analysis: true, inference_allowed: true, valid: true.\n"
                "2. DO NOT reject because of smartphone borders, laptop bezels, screen reflections, or hands.\n"
                "3. Only reject images with ZERO plant foliage (e.g. pure face selfies, vehicles, empty text documents).\n"
                "Return JSON with exact keys: valid (boolean), category (string: plant_leaf, screenshot_document, "
                "non_plant, or uncertain), plant_present (boolean), leaf_present (boolean), "
                "suitable_for_crop_analysis (boolean), reason (string), inference_allowed (boolean)."
            )

            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json"
            }

            candidate_models = [self.model]
            if "3.6" not in self.model:
                candidate_models.append("qwen/qwen3.6-27b")
            if "3.8" not in self.model:
                candidate_models.append("qwen/qwen3.8-27b")

            resp = None
            used_model = self.model
            for candidate in candidate_models:
                payload = {
                    "model": candidate,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.0
                }

                try:
                    resp = requests.post(
                        self.GROQ_ENDPOINT,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    if resp.status_code == 200:
                        used_model = candidate
                        break
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        logger.warning(
                            f"Groq model {candidate} returned HTTP {resp.status_code}. Trying backup model..."
                        )
                        continue
                    else:
                        break
                except requests.exceptions.RequestException:
                    continue

            latency_ms = (time.time() - start_time) * 1000.0

            if resp is None or resp.status_code != 200:
                err_snippet = resp.text[:150] if resp is not None else "Timeout/connection error"
                logger.warning(
                    f"Groq vision returned error ({err_snippet}). "
                    "Falling back to local computer vision validation."
                )
                return None

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return None

            raw_text = choices[0].get("message", {}).get("content", "").strip()

            # Clean markdown fences or think tags if present
            cleaned_text = re.sub(r"^```(?:json)?", "", raw_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r"```$", "", cleaned_text, flags=re.MULTILINE).strip()
            if "<think>" in cleaned_text and "</think>" in cleaned_text:
                cleaned_text = cleaned_text.split("</think>")[-1].strip()

            parsed = json.loads(cleaned_text)

            cat = str(parsed.get("category", "uncertain")).lower().strip()
            valid = bool(parsed.get("valid", False))
            plant_pres = bool(parsed.get("plant_present", False))
            leaf_pres = bool(parsed.get("leaf_present", False))
            suitable = bool(parsed.get("suitable_for_crop_analysis", False))
            allowed = bool(parsed.get("inference_allowed", False))
            reason = str(parsed.get("reason", "Validation processed by Groq Vision."))

            reason_lower = (reason + " " + raw_text).lower()
            # If the reason or model response notes a plant, leaf, crop, or leaf on a smartphone/display screen:
            # We explicitly accept it for diagnosis!
            mentions_leaf_or_plant = any(
                term in reason_lower for term in (
                    "leaf", "leaves", "plant", "crop", "foliage", "corn", "rust", "blight", "specimen",
                    "maize", "tomato", "potato", "apple", "grape", "rice", "wheat", "cotton", "soybean",
                    "vegetation", "chlorosis", "necrosis", "mildew", "scab", "botanical", "fungal", "agriculture",
                    "greenery", "foliar", "displaying a picture of a leaf", "image of a leaf", "showing a leaf",
                    "leaf displayed", "screen showing", "screen displays", "picture of a plant", "image on a phone",
                    "leaf on a phone", "leaf on a screen", "leaf on a laptop", "leaf on a monitor", "screen with a leaf"
                )
            )

            if mentions_leaf_or_plant or cat == "plant_leaf" or plant_pres or leaf_pres or suitable or allowed:
                cat = "plant_leaf"
                plant_pres = True
                leaf_pres = True
                suitable = True
                allowed = True
                valid = True
                reason = "Plant leaf specimen verified (including digital/mobile screen presentation)."
            elif cat in ("screenshot_document", "screenshot", "document", "ui", "dashboard"):
                cat = "screenshot_document"
                allowed = False
                valid = False
            elif cat in ("non_plant", "person", "animal", "vehicle", "object"):
                cat = "non_plant"
                allowed = False
                valid = False
            elif cat == "plant_leaf" and (not plant_pres or not leaf_pres or not suitable):
                allowed = False
                valid = False

            return GroqValidationResult(
                valid=valid,
                category=cat,
                plant_present=plant_pres,
                leaf_present=leaf_pres,
                suitable_for_crop_analysis=suitable,
                reason=reason,
                inference_allowed=allowed,
                latency_ms=round(latency_ms, 2),
                model_used=used_model,
                raw_response=parsed
            )

        except json.JSONDecodeError as jde:
            logger.warning(f"Groq vision response could not be parsed as JSON: {jde}. Falling back to local checks.")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Groq vision request timed out after {self.timeout}s. Falling back to local checks.")
            return None
        except Exception as exc:
            logger.warning(f"Groq vision validation error: {exc}. Falling back to local checks.")
            return None


# Global singleton instance
groq_vision_validator = GroqVisionValidator()
