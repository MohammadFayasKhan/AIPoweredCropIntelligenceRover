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
            # The prompt is strict: Groq must accurately classify the image.
            # It accepts real plant/leaf photos AND plant leaves displayed on screens,
            # but REJECTS app screenshots, documents, notebooks, and non-plant objects.
            system_prompt = (
                "/no_thinking\n"
                "You are a strict agricultural image classifier for a crop disease diagnosis system. "
                "Your job is to determine if the image contains an ACTUAL plant leaf or crop foliage "
                "that can be analyzed for disease.\n\n"
                "ACCEPT (category: plant_leaf) if:\n"
                "- A real plant leaf or crop foliage is the primary subject\n"
                "- A plant leaf is displayed on a phone/tablet/laptop/monitor screen "
                "(the CONTENT on the screen is a plant leaf photo)\n"
                "- A person is holding a real plant leaf\n\n"
                "REJECT as screenshot_document if:\n"
                "- Software UI, app interface, dashboard, web application (even if it has green colors or plant icons/logos)\n"
                "- Documents, PDFs, spreadsheets, text pages\n"
                "- A screenshot of a website or application\n"
                "- A photo of a notebook, book cover, diary, or printed material\n\n"
                "REJECT as non_plant if:\n"
                "- Animals, vehicles, buildings, furniture, electronics, household objects\n"
                "- Human selfies or portraits without any plant leaf\n"
                "- Random objects on a desk or table that are NOT plant leaves\n"
                "- A laptop/computer device itself (not showing a leaf on its screen)\n\n"
                "IMPORTANT: Green color alone does NOT make something a plant. "
                "A green-themed app UI is still a screenshot_document. "
                "A green notebook cover is still non_plant.\n"
                "Output strictly raw JSON without markdown fences."
            )

            user_prompt = (
                "Classify this image into exactly one category. "
                "Return JSON with these exact keys:\n"
                "- valid (boolean): true only if a real plant leaf is present\n"
                "- category (string): exactly one of: plant_leaf, screenshot_document, non_plant, uncertain\n"
                "- plant_present (boolean): true only if an actual plant/crop is visible\n"
                "- leaf_present (boolean): true only if an actual leaf is visible\n"
                "- suitable_for_crop_analysis (boolean): true only if the leaf can be analyzed for disease\n"
                "- reason (string): brief explanation of what you see\n"
                "- inference_allowed (boolean): true only if category is plant_leaf\n\n"
                "Remember: app screenshots with green UI themes are screenshot_document, NOT plant_leaf. "
                "Photos of notebooks, book covers, or printed materials are non_plant. "
                "Only classify as plant_leaf if you can see an ACTUAL plant leaf (real or displayed on a screen)."
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

            # Trust Groq's response directly — no aggressive keyword overrides
            cat = str(parsed.get("category", "uncertain")).lower().strip()
            valid = bool(parsed.get("valid", False))
            plant_pres = bool(parsed.get("plant_present", False))
            leaf_pres = bool(parsed.get("leaf_present", False))
            suitable = bool(parsed.get("suitable_for_crop_analysis", False))
            allowed = bool(parsed.get("inference_allowed", False))
            reason = str(parsed.get("reason", "Validation processed by Groq Vision."))

            # Normalize category names (Groq may return slight variations)
            if cat in ("screenshot", "document", "ui", "dashboard", "screenshot_document"):
                cat = "screenshot_document"
                allowed = False
                valid = False
                plant_pres = False
                leaf_pres = False
                suitable = False
            elif cat in ("non_plant", "person", "animal", "vehicle", "object"):
                cat = "non_plant"
                allowed = False
                valid = False
                plant_pres = False
                leaf_pres = False
                suitable = False
            elif cat == "plant_leaf":
                # Groq explicitly said plant_leaf — trust it
                allowed = True
                valid = True
                plant_pres = True
                leaf_pres = True
                suitable = True

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
