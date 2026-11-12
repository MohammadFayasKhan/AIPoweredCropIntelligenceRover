"""
Image Processing and Validation Utilities for Smart Plant Intelligence System.
Production-grade image validation defending against decompression bombs, corrupt byte streams,
unsupported MIME types, dimension violations, and arbitrary disk writes.
"""

from typing import Tuple, Dict, Any, Optional, List
import io
import base64
import uuid
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from backend.app.config import settings

# Enforce decompression bomb protection at the PIL library level
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


class ImageValidationError(Exception):
    """Custom exception raised when uploaded image fails security or format validation."""
    def __init__(self, message: str, recovery_hint: str):
        super().__init__(message)
        self.message = message
        self.recovery_hint = recovery_hint


def generate_safe_sample_id(original_filename: Optional[str] = None) -> str:
    """Generates an unforgeable, request-safe sample ID avoiding arbitrary disk names."""
    ext = ".jpg"
    if original_filename:
        original_ext = Path(original_filename).suffix.lower()
        if original_ext in settings.ALLOWED_EXTENSIONS:
            ext = original_ext
    return f"specimen_{uuid.uuid4().hex[:12]}{ext}"


def validate_uploaded_image(
    file_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validates uploaded file payload server-side:
    1. Rejection of empty or sub-minimum payloads (< 2 KB).
    2. Byte length ceiling constraint (<= 15 MB).
    3. MIME type and extension validation.
    4. Magic header byte inspection (JPEG, PNG, WebP).
    5. PIL decompression integrity and decompression-bomb defenses.
    6. OpenCV BGR color channel decompression.
    7. Pixel dimension boundaries (min 64x64, max 4096x4096).
    """
    safe_filename = filename or "uploaded_leaf.jpg"
    file_size = len(file_bytes)

    if file_size < settings.MIN_UPLOAD_SIZE_BYTES:
        raise ImageValidationError(
            message=f"Uploaded file '{safe_filename}' is too small ({file_size} bytes).",
            recovery_hint="Please provide a valid, sharp photograph of a plant leaf (minimum 2 KB)."
        )

    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        curr_mb = file_size / (1024 * 1024)
        raise ImageValidationError(
            message=f"Uploaded file exceeds maximum allowed limit of {max_mb:.1f} MB (received {curr_mb:.1f} MB).",
            recovery_hint="Please compress or resize the photograph to under 15 MB before uploading."
        )

    # Optional content-type check if provided
    if content_type:
        clean_ct = content_type.lower().split(";")[0].strip()
        if clean_ct and clean_ct not in settings.ALLOWED_MIME_TYPES:
            raise ImageValidationError(
                message=f"Unsupported content type '{clean_ct}'.",
                recovery_hint="Supported formats are JPEG (.jpg, .jpeg), PNG (.png), and WebP (.webp)."
            )

    # Magic byte verification
    is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
    is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = file_bytes.startswith(b"RIFF") and b"WEBP" in file_bytes[:16]

    if not (is_jpeg or is_png or is_webp):
        raise ImageValidationError(
            message=f"Unsupported file format or invalid magic bytes for '{safe_filename}'.",
            recovery_hint="Supported formats are JPEG (.jpg, .jpeg), PNG (.png), and WebP (.webp)."
        )

    # Decode check via PIL with decompression bomb trap and EXIF orientation normalization
    try:
        from PIL import ImageOps
        pil_raw = Image.open(io.BytesIO(file_bytes))
        # Transpose according to EXIF orientation tag so phone photos are not sideways/upside-down
        pil_img = ImageOps.exif_transpose(pil_raw)
        if pil_img is None:
            pil_img = pil_raw
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Image.DecompressionBombError:
        raise ImageValidationError(
            message="Image exceeds maximum pixel threshold (decompression bomb protection triggered).",
            recovery_hint="Please rescale image dimensions to under 4096x4096 pixels."
        )
    except Exception as e:
        raise ImageValidationError(
            message=f"Image file header or byte stream is corrupted: {str(e)}",
            recovery_hint="The file could not be parsed. Please export or recapture the leaf photograph."
        )

    if img_bgr is None or img_bgr.size == 0:
        raise ImageValidationError(
            message="Image decompression yielded empty pixel buffer.",
            recovery_hint="Ensure the image has valid color channels and is not an empty or corrupt stream."
        )

    h, w, c = img_bgr.shape
    if w < settings.MIN_IMAGE_DIMENSION or h < settings.MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            message=f"Image resolution ({w}x{h}) is below the minimum required resolution ({settings.MIN_IMAGE_DIMENSION}x{settings.MIN_IMAGE_DIMENSION}).",
            recovery_hint="Please capture the leaf closer to the camera lens to provide sufficient foliar detail."
        )

    if w > settings.MAX_IMAGE_DIMENSION or h > settings.MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            message=f"Image resolution ({w}x{h}) exceeds maximum supported resolution ({settings.MAX_IMAGE_DIMENSION}x{settings.MAX_IMAGE_DIMENSION}).",
            recovery_hint=f"Please downscale oversized images to within {settings.MAX_IMAGE_DIMENSION}x{settings.MAX_IMAGE_DIMENSION} before uploading."
        )

    # Perform botanical quality assessment (blur, illumination, and foliar coverage)
    from backend.app.utils.image_quality import assess_image_quality
    quality_assessment = assess_image_quality(img_bgr)

    meta = {
        "width": w,
        "height": h,
        "channels": c,
        "size_bytes": file_size,
        "format": "JPEG" if is_jpeg else ("PNG" if is_png else "WebP"),
        "image_quality": quality_assessment
    }

    return img_bgr, meta


def preprocess_tier1(img_bgr: np.ndarray, device: torch.device, input_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Tier 1 Preprocessing:
    Converts BGR to RGB, resizes to target input_size (256x256 for EfficientNetV2-S as per preprocessing_config.json),
    applies standard ImageNet mean/std normalization.
    Returns Float32 Tensor [1, 3, H, W] on target compute device.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (img_resized - mean) / std

    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def preprocess_tier3(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Tier 3 Preprocessing:
    Converts BGR to RGB, resizes to 256x256 for Mobile-UNet, applies ImageNet normalization.
    Returns Float32 Tensor [1, 3, 256, 256] on target compute device.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (img_resized - mean) / std

    tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def create_segmentation_overlay_base64(orig_w: int, orig_h: int, pred_mask_256: np.ndarray) -> str:
    """
    Generates a transparent RGBA PNG overlay matching original image dimensions.
    - Healthy foliar tissue (class == 1) is tinted translucent soft green (#52B788, alpha 70)
    - Lesion focus areas (class == 2) are mapped to translucent crimson (#E63946, alpha 195)
    - Background (class == 0) remains 100% transparent.
    Returns a data URI string: 'data:image/png;base64,...'.
    """
    mask_resized = cv2.resize(pred_mask_256.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    leaf_indices = (mask_resized == 1)
    lesion_indices = (mask_resized == 2)

    rgba[leaf_indices] = [40, 167, 69, 70]    # Soft green for healthy foliage
    rgba[lesion_indices] = [230, 57, 70, 195]  # Crimson for active lesions

    success, encoded_png = cv2.imencode(".png", rgba)
    if not success:
        return ""

    b64_str = base64.b64encode(encoded_png).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def create_segmentation_mask_base64(orig_w: int, orig_h: int, pred_mask_256: np.ndarray) -> str:
    """
    Generates an opaque RGB PNG representation of the foliar segmentation mask:
    - Background (class 0): Dark charcoal [18, 22, 20]
    - Healthy Foliage (class 1): Botanical green [46, 125, 50]
    - Foliar Lesion (class 2): High-contrast crimson [229, 57, 53]
    Returns a data URI string: 'data:image/png;base64,...'.
    """
    mask_resized = cv2.resize(pred_mask_256.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    rgb = np.full((orig_h, orig_w, 3), 20, dtype=np.uint8)

    rgb[mask_resized == 1] = [50, 140, 60]   # Foliar canopy
    rgb[mask_resized == 2] = [230, 50, 60]   # Lesion foci

    success, encoded_png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        return ""

    b64_str = base64.b64encode(encoded_png).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def scale_boxes_to_original(boxes_xyxy: List[List[float]], model_w: int, model_h: int, orig_w: int, orig_h: int) -> List[List[int]]:
    """
    Scales bounding box coordinates from detection space to original image space,
    preserving exact pixel alignment.
    """
    scaled: List[List[int]] = []
    scale_x = orig_w / float(model_w)
    scale_y = orig_h / float(model_h)

    for box in boxes_xyxy:
        x1 = int(round(box[0] * scale_x))
        y1 = int(round(box[1] * scale_y))
        x2 = int(round(box[2] * scale_x))
        y2 = int(round(box[3] * scale_y))
        # Clamp to bounds
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(x1 + 1, min(x2, orig_w))
        y2 = max(y1 + 1, min(y2, orig_h))
        scaled.append([x1, y1, x2, y2])
    return scaled


def reverse_letterbox_xyxy(
    boxes_xyxy: List[List[float]],
    input_w: int,
    input_h: int,
    orig_w: int,
    orig_h: int,
) -> List[List[int]]:
    """
    Reverses YOLO aspect-preserving letterbox padding and scaling back to original image pixels.
    """
    gain = min(float(input_h) / float(max(1, orig_h)), float(input_w) / float(max(1, orig_w)))
    pad_w = (float(input_w) - float(orig_w) * gain) / 2.0
    pad_h = (float(input_h) - float(orig_h) * gain) / 2.0

    mapped: List[List[int]] = []
    for box in boxes_xyxy:
        x1 = int(round(max(0.0, min((float(box[0]) - pad_w) / gain, float(orig_w - 1)))))
        y1 = int(round(max(0.0, min((float(box[1]) - pad_h) / gain, float(orig_h - 1)))))
        x2 = int(round(max(float(x1 + 1), min((float(box[2]) - pad_w) / gain, float(orig_w)))))
        y2 = int(round(max(float(y1 + 1), min((float(box[3]) - pad_h) / gain, float(orig_h)))))
        mapped.append([x1, y1, x2, y2])
    return mapped


def scale_segmentation_mask_to_original(
    mask: np.ndarray,
    orig_w: int,
    orig_h: int,
    letterbox_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Maps a binary or class-indexed YOLO26-Seg mask cleanly back into exact original (orig_h, orig_w) dimensions,
    unpadding letterbox borders when letterbox_shape=(in_w, in_h) is supplied.
    """
    m_arr = mask.astype(np.uint8)
    if letterbox_shape is not None:
        in_w, in_h = letterbox_shape
        m_full = cv2.resize(m_arr, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
        gain = min(float(in_h) / float(max(1, orig_h)), float(in_w) / float(max(1, orig_w)))
        pad_w = int(round((float(in_w) - float(orig_w) * gain) / 2.0))
        pad_h = int(round((float(in_h) - float(orig_h) * gain) / 2.0))
        top, left = max(0, pad_h), max(0, pad_w)
        bottom = max(top + 1, in_h - top)
        right = max(left + 1, in_w - left)
        m_arr = m_full[top:bottom, left:right]
    return cv2.resize(m_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

