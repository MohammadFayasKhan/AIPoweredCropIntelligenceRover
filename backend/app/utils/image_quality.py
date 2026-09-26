"""
Image Quality Assessment & Foliar Preflight Module for AI Powered Crop Intelligence Rover.
Evaluates uploaded leaf photography before expensive neural inference:
  • Laplacian variance for focal and motion blur detection
  • Illumination mean and dynamic range for exposure extremes
  • Excess Green Index (ExG) and color space analysis for botanical presence
  • Actionable feedback and recapture guidance to avoid false high-confidence predictions

Flow:
  Raw BGR Array → Grayscale Laplacian → Photometric Analysis → Vegetation Index → Quality Score
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


def assess_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Executes a comprehensive botanical image-quality audit on raw BGR leaf photography.

    Mathematical Basis:
      1. Blur: Laplacian variance Var(∇²I) measures high-frequency edge energy.
         ⤷ Low variance (< 45.0) indicates motion blur or out-of-focus camera capture.
      2. Illumination: Mean pixel intensity μ and standard deviation σ.
         ⤷ Severe underexposure (μ < 32.0) or overexposure (μ > 228.0) hides pathology symptoms.
         ⤷ Flat contrast (σ < 22.0) obscures subtle foliar lesion margins.
      3. Botanical Presence: Excess Green Index ExG = 2G - R - B and HSV hue thresholding.
         ⤷ Distinguishes actual plant leaves from non-agricultural objects or plain backgrounds.

    Returns:
      Dictionary matching the ImageQualityAssessment schema.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Blur evaluation via Laplacian edge variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Photometric distribution (mean brightness and contrast dynamic range)
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    # 3. Botanical / foliar tissue presence analysis
    # Convert to RGB to calculate Excess Green Index (ExG = 2G - R - B)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    exg = 2.0 * g_ch - r_ch - b_ch

    # Foliar mask: pixels where green exceeds red and blue, or typical leaf HSV range
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    # Green and yellow-green foliar hues span hue 25 to 95 with moderate saturation
    green_mask = ((hue >= 25) & (hue <= 95) & (sat >= 30)) | (exg > 15.0)
    # Also capture brown / necrotic rust lesions (hue 10 to 25 with saturation)
    necrotic_mask = (hue >= 8) & (hue <= 25) & (sat >= 40)
    foliar_mask = green_mask | necrotic_mask

    foliar_pixel_count = int(np.sum(foliar_mask))
    total_pixels = float(h * w)
    greenness_ratio = float(foliar_pixel_count / max(1.0, total_pixels))

    # 4. Defect identification & scoring
    quality_issues: List[str] = []
    is_acceptable = True
    guidance_parts: List[str] = []

    # Check blur
    if blur_score < 40.0:
        quality_issues.append("Severe motion or focal blur detected")
        guidance_parts.append("Hold the camera steady or tap to focus directly on the leaf surface.")
        is_acceptable = False
    elif blur_score < 70.0:
        quality_issues.append("Mild blur present")
        guidance_parts.append("Ensure the lens is focused sharply on foliar symptoms.")

    # Check illumination
    if brightness_mean < 30.0:
        quality_issues.append("Critically underexposed (too dark)")
        guidance_parts.append("Capture the specimen in brighter, indirect daylight or use a diffuse light source.")
        is_acceptable = False
    elif brightness_mean > 230.0:
        quality_issues.append("Critically overexposed (washed out glare)")
        guidance_parts.append("Avoid direct sun glare; shield the leaf or capture under gentle shade.")
        is_acceptable = False
    elif brightness_mean < 55.0:
        quality_issues.append("Sub-optimal lighting (dim)")
    elif brightness_mean > 205.0:
        quality_issues.append("High brightness with possible reflection")

    # Check contrast
    if contrast_std < 20.0:
        quality_issues.append("Extremely low contrast dynamic range")
        guidance_parts.append("Adjust lighting angle so lesion margins stand out from healthy leaf tissue.")
        if len(quality_issues) > 1:
            is_acceptable = False

    # Check foliar content
    if greenness_ratio < 0.05:
        quality_issues.append("No clear plant foliage or canopy detected")
        guidance_parts.append("Center a single leaf specimen within the camera frame.")
        is_acceptable = False
    elif greenness_ratio < 0.15:
        quality_issues.append("Low foliar canopy coverage in frame")
        guidance_parts.append("Move closer so the plant leaf fills at least 50% of the viewfinder.")

    # Compute continuous composite quality score [0.0, 1.0]
    # Sharpness component (normalized to [0, 1] with asymptote at 500)
    norm_sharpness = min(1.0, blur_score / 350.0)
    # Brightness penalty (optimal around 110-150)
    bright_dist = abs(brightness_mean - 130.0) / 130.0
    norm_brightness = max(0.0, 1.0 - bright_dist)
    # Foliar canopy component
    norm_foliar = min(1.0, greenness_ratio / 0.40)

    quality_score = float(np.clip(
        0.45 * norm_sharpness + 0.30 * norm_brightness + 0.25 * norm_foliar,
        0.0,
        1.0
    ))

    # Determine qualitative level (Good, Acceptable, Poor)
    if quality_score >= 0.70 and is_acceptable and len(quality_issues) == 0:
        quality_level = "Good"
        summary_text = "Good image quality: specimen is sharply focused and well-illuminated for reliable botanical analysis."
    elif is_acceptable and quality_score >= 0.42:
        quality_level = "Acceptable"
        if quality_issues:
            summary_text = f"Acceptable image quality: {'; '.join(quality_issues)}. Analysis may proceed with standard caution."
        else:
            summary_text = "Acceptable image quality: adequate for diagnostic inference."
    else:
        quality_level = "Poor"
        if quality_issues:
            summary_text = f"Poor image quality: {'; '.join(quality_issues)}. Retaking the photograph is strongly recommended."
        else:
            summary_text = "Poor image quality: insufficient foliar detail for confident diagnosis."

    recapture_guidance = " ".join(guidance_parts) if guidance_parts else "Image satisfies quality thresholds for diagnostic inference."

    return {
        "quality_score": round(quality_score, 3),
        "quality_level": quality_level,
        "summary_text": summary_text,
        "is_acceptable": is_acceptable,
        "is_usable": is_acceptable,
        "blur_score": round(blur_score, 1),
        "brightness_mean": round(brightness_mean, 1),
        "contrast_std": round(contrast_std, 1),
        "greenness_ratio": round(greenness_ratio, 3),
        "quality_issues": quality_issues,
        "warnings": quality_issues,
        "recapture_guidance": recapture_guidance,
        "recommendation": recapture_guidance,
    }

