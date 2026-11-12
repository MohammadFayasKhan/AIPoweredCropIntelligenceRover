#!/usr/bin/env python3
"""
prepare_yolo26_tridataset.py
----------------------------
Strictly non-destructive dataset discovery, license compliance, deduplication,
validation, and task-separated YOLO26 preparation pipeline.

Writes ONLY under the specified output directory (default: artifacts/yolo26_v1/datasets)
and NEVER modifies existing datasets, manifests, configs, or checkpoints.

Strict Scientific Integrity Rules:
  1. Whole-leaf Specimen Detection (YOLO26 Detect): uses ONLY PlantDoc Pascal VOC
     bounding box annotations in canonical 29-class alphabetical order.
  2. Foliar Lesion Segmentation (YOLO26-Seg / Semantic): uses ONLY genuine human-annotated
     PlantSeg masks/polygons if discovered in input roots. Explicitly rejects and
     excludes HSV/LAB color-threshold pseudo-labels (yolo_lesions, processed/segmentation).
  3. Classification-Only Datasets (PlantVillage, PlantWild): NEVER converted into
     synthetic bounding boxes or pseudo-masks; used solely to compile healthy
     negative-control & domain-shift evaluation indices with full license tracking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Canonical alphabetical 29-class PlantDoc vocabulary matching cv/configs/yolo_plantdoc.yaml
CANONICAL_PLANTDOC_29_CLASSES: List[str] = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf",
    "grape leaf black rot",
]

PLANTDOC_CLASS_LOWER_TO_CANONICAL: Dict[str, str] = {
    c.lower(): c for c in CANONICAL_PLANTDOC_29_CLASSES
}
PLANTDOC_CLASS_TO_ID: Dict[str, int] = {
    c: idx for idx, c in enumerate(CANONICAL_PLANTDOC_29_CLASSES)
}

DATASET_LICENSES: Dict[str, Dict[str, Any]] = {
    "PlantVillage": {
        "citation": "Hughes & Salathe (2015), arXiv:1511.08060",
        "license": "CC0 1.0 Universal (Public Domain)",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "spatial_modality": "classification_only",
        "notes": "Zero genuine spatial bounding boxes or human lesion masks.",
    },
    "PlantDoc": {
        "citation": "Singh et al. (2019), ACM CoNEXT",
        "license": "MIT License",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "spatial_modality": "object_detection_pascal_voc",
        "notes": "29 foliar leaf/disease whole-specimen Pascal VOC bounding box classes.",
    },
    "PlantWild": {
        "citation": "Wei, Chen, Huang & Yu (2024), ACM Multimedia",
        "license": "CC BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives)",
        "commercial_use_allowed": False,
        "redistribution_allowed": False,
        "spatial_modality": "classification_only",
        "notes": "RESTRICTED: Non-commercial academic/research only; derivatives/redistribution prohibited. Zero spatial annotations.",
    },
    "PlantSeg": {
        "citation": "Wei et al. (2024), PlantSeg: Large-Scale In-the-wild Dataset for Plant Disease Segmentation",
        "license": "Research/Academic Zenodo Open Access (verify repository LICENSE before redistribution)",
        "commercial_use_allowed": False,
        "redistribution_allowed": False,
        "spatial_modality": "lesion_segmentation",
        "notes": "Genuine pixel-level LabelMe JSON polygons and PNG disease segmentation masks when mounted.",
    },
}


def compute_hashes(img_path: Path, hash_size: int = 8) -> Tuple[str, str]:
    """Computes exact MD5 and 64-bit perceptual dHash for leakage and duplicate detection."""
    raw = img_path.read_bytes()
    md5_hex = hashlib.md5(raw).hexdigest()
    try:
        with Image.open(img_path) as im:
            im.load()
            gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            arr = np.asarray(gray)
            diff = arr[:, 1:] > arr[:, :-1]
            dhash_hex = hashlib.md5(diff.tobytes()).hexdigest()[:16]
    except Exception:
        dhash_hex = ""
    return md5_hex, dhash_hex


def validate_image_file(img_path: Path, min_dim: int = 32) -> Optional[Tuple[int, int]]:
    """Verifies image readability and minimum pixel dimensions; returns (width, height) or None."""
    try:
        if not img_path.is_file() or img_path.stat().st_size < 256:
            return None
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im2:
            w, h = im2.size
        if w < min_dim or h < min_dim:
            return None
        return int(w), int(h)
    except Exception:
        return None


def _find_matching_image(stem_path: Path, fallback_filename: Optional[str] = None) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"):
        cand = stem_path.with_suffix(ext)
        if cand.exists():
            return cand
    if fallback_filename:
        cand = stem_path.parent / Path(fallback_filename).name
        if cand.exists():
            return cand
    return None


def _dir_has_file_ext(directory: Path, exts: Tuple[str, ...]) -> bool:
    """Fast O(entries) check for any matching file inside directory or 1-2 nested subdirs."""
    candidates = [directory]
    for sub in ("train", "TRAIN", "test", "TEST", "val", "VAL", "annotations", "masks", "images"):
        sub_p = directory / sub
        if sub_p.is_dir():
            candidates.append(sub_p)
            try:
                with os.scandir(sub_p) as sit:
                    for e in sit:
                        if e.is_dir() and not e.name.startswith("."):
                            candidates.append(Path(e.path))
            except OSError:
                pass
    exts_low = tuple(e.lower() for e in exts)
    for cand in candidates:
        try:
            with os.scandir(cand) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(exts_low):
                        return True
        except OSError:
            continue
    return False


def _scan_roots_fast(search_roots: List[Path], max_depth: int = 9) -> Dict[str, Optional[Path]]:
    """Breadth-first os.walk discovery pruning massive image leaf directories."""
    discovered: Dict[str, Optional[Path]] = {
        "plantdoc_root": None,
        "plantseg_root": None,
        "plantvillage_root": None,
        "plantwild_root": None,
    }
    prune_dirs = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "processed", "yolo_lesions", "segmentation", "runs", "weights",
        "objects", "images", "masks", "labels", "classes", "color",
        "train", "val", "test", "train2017", "val2017",
    }

    for root in search_roots:
        if not root.exists():
            continue
        root_resolved = root.resolve()
        for dirpath, dirnames, _ in os.walk(root, followlinks=True):
            p = Path(dirpath)
            try:
                depth = len(p.resolve().relative_to(root_resolved).parts)
            except Exception:
                depth = 0
            dir_low = p.name.lower()
            low_parts = {pt.lower() for pt in p.parts}

            if "processed" not in low_parts and "yolo_lesions" not in low_parts:
                # 1. PlantDoc VOC
                if discovered["plantdoc_root"] is None and (
                    dir_low in {"plantdoc_od_repo", "plantdoc_repo", "plantdoc-dataset", "plantdoc"}
                    or "plantdoc" in dir_low
                ):
                    if _dir_has_file_ext(p, (".xml",)):
                        discovered["plantdoc_root"] = p
                        dirnames[:] = []
                        continue

                # 2. PlantSeg genuine lesion masks
                if discovered["plantseg_root"] is None and "plantseg" in dir_low:
                    has_imgs = (p / "images").is_dir() or _dir_has_file_ext(p, (".jpg", ".jpeg", ".png"))
                    has_ann = (
                        (p / "annotations").is_dir()
                        or (p / "masks").is_dir()
                        or _dir_has_file_ext(p, (".json",))
                    )
                    if has_imgs and has_ann:
                        discovered["plantseg_root"] = p
                        dirnames[:] = []
                        continue

                # 3. PlantVillage color directory
                if discovered["plantvillage_root"] is None and "plantvillage" in dir_low:
                    color_sub = p / "raw" / "color"
                    if color_sub.is_dir():
                        discovered["plantvillage_root"] = color_sub
                        dirnames[:] = []
                        continue
                    if any("___" in d for d in dirnames):
                        discovered["plantvillage_root"] = p
                        dirnames[:] = []
                        continue

                # 4. PlantWild directory
                if discovered["plantwild_root"] is None and "plantwild" in dir_low:
                    img_sub = p / "plantwild" / "images"
                    if img_sub.is_dir():
                        discovered["plantwild_root"] = img_sub
                    elif (p / "images").is_dir():
                        discovered["plantwild_root"] = p / "images"
                    else:
                        discovered["plantwild_root"] = p
                    dirnames[:] = []
                    continue

            if depth >= max_depth or all(v is not None for v in discovered.values()):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(
                d for d in dirnames
                if d.lower() not in prune_dirs and not d.startswith(".")
            )

    return discovered


def discover_tri_datasets(
    search_roots: List[Path], extract_dir: Optional[Path] = None
) -> Dict[str, Optional[Path]]:
    """
    Fast directory-pruned discovery of PlantDoc, PlantSeg, PlantVillage, and PlantWild without
    modifying input directories. Explicitly ignores synthetic HSV/LAB directories.
    If extract_dir is provided and required spatial sets are packed inside ZIP archives under
    search_roots, safely extracts matching archives into extract_dir (read-only on inputs).
    """
    discovered = _scan_roots_fast(search_roots)

    if extract_dir is not None and (
        discovered["plantdoc_root"] is None or discovered["plantseg_root"] is None
    ):
        archive_candidates: List[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            root_resolved = root.resolve()
            for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
                p = Path(dirpath)
                try:
                    depth = len(p.resolve().relative_to(root_resolved).parts)
                except Exception:
                    depth = 0
                for fn in filenames:
                    low = fn.lower()
                    if low.endswith(".zip") and any(
                        tok in low for tok in ("plantdoc", "plantseg", "cv_core_assets", "tridataset", "smartcrop")
                    ):
                        archive_candidates.append(p / fn)
                if depth >= 5:
                    dirnames[:] = []

        if archive_candidates:
            extract_dir.mkdir(parents=True, exist_ok=True)
            ext_root_res = extract_dir.resolve()
            for zpath in archive_candidates[:4]:
                target_sub = extract_dir / zpath.stem
                if not target_sub.exists():
                    try:
                        with zipfile.ZipFile(zpath, "r") as zf:
                            for member in zf.infolist():
                                dest = (target_sub / member.filename).resolve()
                                if str(dest).startswith(str(ext_root_res) + os.sep) or dest == ext_root_res:
                                    zf.extract(member, target_sub)
                    except Exception:
                        continue
            re_discovered = _scan_roots_fast([extract_dir], max_depth=6)
            for k, v in re_discovered.items():
                if discovered[k] is None and v is not None:
                    discovered[k] = v

    return discovered


def parse_plantdoc_voc_xml(
    xml_path: Path, img_w: int, img_h: int
) -> Tuple[List[Tuple[int, float, float, float, float]], int, int]:
    """
    Parses a PlantDoc Pascal VOC XML file and validates every bounding box against image dimensions.
    Returns: (valid YOLO boxes, invalid_box_count, unknown_class_count).
    """
    valid_boxes: List[Tuple[int, float, float, float, float]] = []
    invalid_boxes = 0
    unknown_classes = 0

    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return [], 1, 0

    for obj in root.findall("object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if name_el is None or name_el.text is None or bnd is None:
            invalid_boxes += 1
            continue

        raw_cls = name_el.text.strip()
        canon_cls = PLANTDOC_CLASS_LOWER_TO_CANONICAL.get(raw_cls.lower())
        if canon_cls is None:
            unknown_classes += 1
            continue
        cid = PLANTDOC_CLASS_TO_ID[canon_cls]

        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
        except Exception:
            invalid_boxes += 1
            continue

        xmin = max(0.0, min(xmin, float(img_w - 1)))
        ymin = max(0.0, min(ymin, float(img_h - 1)))
        xmax = max(0.0, min(xmax, float(img_w)))
        ymax = max(0.0, min(ymax, float(img_h)))

        if (xmax - xmin) < 2.0 or (ymax - ymin) < 2.0:
            invalid_boxes += 1
            continue

        xc = ((xmin + xmax) / 2.0) / float(img_w)
        yc = ((ymin + ymax) / 2.0) / float(img_h)
        bw = (xmax - xmin) / float(img_w)
        bh = (ymax - ymin) / float(img_h)
        valid_boxes.append((cid, xc, yc, bw, bh))

    return valid_boxes, invalid_boxes, unknown_classes


def prepare_plantdoc_yolo26(
    plantdoc_root: Optional[Path],
    out_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Converts PlantDoc Pascal VOC XMLs into leakage-free train / val / test YOLO detection splits.
    Preserves official PlantDoc test split cleanly and removes exact/perceptual duplicates from train/val.
    """
    ds_dir = out_dir / "plantdoc_yolo26"
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "status": "MISSING",
        "dataset": "PlantDoc",
        "license": DATASET_LICENSES["PlantDoc"]["license"],
        "classes_count": len(CANONICAL_PLANTDOC_29_CLASSES),
        "splits": {"train": 0, "val": 0, "test": 0},
        "boxes": {"train": 0, "val": 0, "test": 0},
        "small_boxes_under_1pct": {"train": 0, "val": 0, "test": 0},
        "corrupt_images_skipped": 0,
        "invalid_boxes_skipped": 0,
        "unknown_classes_skipped": 0,
        "cross_split_duplicates_removed": 0,
        "intra_split_duplicates_removed": 0,
        "class_box_distribution": {},
        "yaml_path": None,
    }

    if plantdoc_root is None or not plantdoc_root.exists():
        return summary

    xml_files = sorted(plantdoc_root.rglob("*.xml"))
    if not xml_files:
        return summary

    official_test_records: List[Dict[str, Any]] = []
    train_pool_records: List[Dict[str, Any]] = []

    for xml_p in xml_files:
        fn_el = None
        try:
            fn_el = ET.parse(xml_p).getroot().find("filename")
        except Exception:
            pass
        fallback_fn = fn_el.text.strip() if (fn_el is not None and fn_el.text) else None
        img_p = _find_matching_image(xml_p, fallback_filename=fallback_fn)
        if img_p is None:
            summary["corrupt_images_skipped"] += 1
            continue

        dims = validate_image_file(img_p)
        if dims is None:
            summary["corrupt_images_skipped"] += 1
            continue
        img_w, img_h = dims

        boxes, inv_cnt, unk_cnt = parse_plantdoc_voc_xml(xml_p, img_w, img_h)
        summary["invalid_boxes_skipped"] += inv_cnt
        summary["unknown_classes_skipped"] += unk_cnt
        if not boxes:
            continue

        md5_hex, dhash_hex = compute_hashes(img_p)
        parts_lower = {p.lower() for p in xml_p.parts} | {p.lower() for p in img_p.parts}
        is_test = "test" in parts_lower and "train" not in parts_lower

        rec = {
            "img_path": img_p,
            "xml_path": xml_p,
            "img_w": img_w,
            "img_h": img_h,
            "boxes": boxes,
            "primary_cls": boxes[0][0],
            "md5": md5_hex,
            "dhash": dhash_hex,
        }
        if is_test:
            official_test_records.append(rec)
        else:
            train_pool_records.append(rec)

    # Deduplicate test first, then prune train pool against test + itself
    seen_test_md5, seen_test_dhash = set(), set()
    clean_test: List[Dict[str, Any]] = []
    for rec in official_test_records:
        if rec["md5"] in seen_test_md5 or (rec["dhash"] and rec["dhash"] in seen_test_dhash):
            summary["intra_split_duplicates_removed"] += 1
            continue
        seen_test_md5.add(rec["md5"])
        if rec["dhash"]:
            seen_test_dhash.add(rec["dhash"])
        clean_test.append(rec)

    seen_train_md5, seen_train_dhash = set(), set()
    clean_train_pool: List[Dict[str, Any]] = []
    for rec in train_pool_records:
        if rec["md5"] in seen_test_md5 or (rec["dhash"] and rec["dhash"] in seen_test_dhash):
            summary["cross_split_duplicates_removed"] += 1
            continue
        if rec["md5"] in seen_train_md5 or (rec["dhash"] and rec["dhash"] in seen_train_dhash):
            summary["intra_split_duplicates_removed"] += 1
            continue
        seen_train_md5.add(rec["md5"])
        if rec["dhash"]:
            seen_train_dhash.add(rec["dhash"])
        clean_train_pool.append(rec)

    if len(clean_train_pool) >= 10:
        cls_counts = Counter(r["primary_cls"] for r in clean_train_pool)
        strat_labels = [
            r["primary_cls"] if cls_counts[r["primary_cls"]] >= 2 else -1
            for r in clean_train_pool
        ]
        try:
            train_recs, val_recs = train_test_split(
                clean_train_pool, test_size=val_ratio, random_state=seed, stratify=strat_labels
            )
        except Exception:
            train_recs, val_recs = train_test_split(
                clean_train_pool, test_size=val_ratio, random_state=seed
            )
    else:
        train_recs = clean_train_pool
        val_recs = []

    cls_dist: Counter = Counter()
    split_map = {"train": train_recs, "val": val_recs, "test": clean_test}
    for sp_name, recs in split_map.items():
        for idx, rec in enumerate(recs):
            stem = f"{sp_name}_{idx:05d}_{rec['img_path'].stem[:40]}"
            stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem)
            dst_img = ds_dir / "images" / sp_name / f"{stem}{rec['img_path'].suffix.lower()}"
            dst_lbl = ds_dir / "labels" / sp_name / f"{stem}.txt"
            shutil.copy2(rec["img_path"], dst_img)

            lbl_lines = []
            for cid, xc, yc, bw, bh in rec["boxes"]:
                lbl_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                cls_dist[CANONICAL_PLANTDOC_29_CLASSES[cid]] += 1
                summary["boxes"][sp_name] += 1
                if (bw * bh) < 0.01:
                    summary["small_boxes_under_1pct"][sp_name] += 1
            dst_lbl.write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")
            summary["splits"][sp_name] += 1

    yaml_path = ds_dir / "yolo26_plantdoc.yaml"
    yaml_lines = [
        f"path: {ds_dir.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if summary["splits"]["test"] > 0:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    for cid, cname in enumerate(CANONICAL_PLANTDOC_29_CLASSES):
        yaml_lines.append(f"  {cid}: '{cname}'")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary["class_box_distribution"] = dict(sorted(cls_dist.items(), key=lambda kv: -kv[1]))
    summary["yaml_path"] = str(yaml_path)
    summary["status"] = "READY" if summary["splits"]["train"] > 0 else "EMPTY"
    return summary


def mask_to_yolo_polygons(
    mask_arr: np.ndarray, class_id: int = 0, min_area_px: float = 16.0
) -> List[Tuple[int, List[float]]]:
    """
    Converts a discrete binary or class-indexed lesion mask into normalized YOLO-Seg polygons
    without modifying mask precision beyond Douglas-Peucker contour simplification.
    """
    h, w = mask_arr.shape[:2]
    if h <= 0 or w <= 0:
        return []
    bin_mask = (mask_arr > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[Tuple[int, List[float]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px or len(cnt) < 3:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(0.75, 0.004 * peri), True)
        if len(approx) < 3:
            approx = cnt
        if len(approx) < 3:
            continue
        pts: List[float] = []
        for pt in approx.reshape(-1, 2):
            nx = max(0.0, min(1.0, float(pt[0]) / float(w)))
            ny = max(0.0, min(1.0, float(pt[1]) / float(h)))
            pts.extend([nx, ny])
        if len(pts) >= 6:
            polygons.append((class_id, pts))
    return polygons


def prepare_plantseg_lesion_yolo26(
    plantseg_root: Optional[Path],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Audits and converts genuine human-annotated PlantSeg lesion masks or LabelMe JSON polygons
    into YOLO26-Seg format (or reports MISSING honestly when PlantSeg is absent).
    Never consumes HSV/LAB pseudo-annotations.
    """
    summary: Dict[str, Any] = {
        "status": "MISSING_PLANTSEG_NO_PSEUDO_FALLBACK",
        "dataset": "PlantSeg",
        "license": DATASET_LICENSES["PlantSeg"]["license"],
        "annotation_modality": None,
        "recommended_yolo26_task": "yolo26s-seg.pt",
        "splits": {"train": 0, "val": 0, "test": 0},
        "instances": {"train": 0, "val": 0, "test": 0},
        "yaml_path": None,
        "note": (
            "PlantSeg genuine human-annotated lesion masks were not found under search roots. "
            "Existing yolo_lesions and processed/segmentation datasets are HSV/LAB pseudo-annotations "
            "and are strictly excluded from YOLO26 lesion training."
        ),
    }
    if plantseg_root is None or not plantseg_root.exists():
        return summary

    ds_dir = out_dir / "plantseg_yolo26_seg"
    for sp in ("train", "val", "test"):
        (ds_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ds_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    json_anns = sorted(plantseg_root.rglob("*.json"))
    png_masks = [
        p for p in sorted(plantseg_root.rglob("*.png"))
        if any(k in {pt.lower() for pt in p.parts} for k in ("mask", "masks", "annotation", "annotations", "gt"))
    ]

    if json_anns:
        summary["annotation_modality"] = "labelme_instance_polygons"
        all_labels: set = set()
        parsed_entries: List[Tuple[Path, int, int, str, List[Tuple[str, List[float]]]]] = []
        for jp in json_anns:
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                shapes = data.get("shapes", [])
                img_fn = data.get("imagePath")
                img_p = _find_matching_image(jp, img_fn)
                if img_p is None:
                    img_candidates = list(plantseg_root.rglob(f"{jp.stem}.*"))
                    img_p = next((c for c in img_candidates if c.suffix.lower() in IMG_EXTS), None)
                if img_p is None:
                    continue
                dims = validate_image_file(img_p)
                if dims is None:
                    continue
                w, h = dims
                parts_l = {pt.lower() for pt in jp.parts} | {pt.lower() for pt in img_p.parts}
                split = "test" if "test" in parts_l else ("val" if "val" in parts_l else "train")
                polys: List[Tuple[str, List[float]]] = []
                for sh in shapes:
                    lbl = str(sh.get("label", "lesion")).strip() or "lesion"
                    pts_raw = sh.get("points", [])
                    if len(pts_raw) < 3:
                        continue
                    norm_pts: List[float] = []
                    for x, y in pts_raw:
                        norm_pts.extend([max(0.0, min(1.0, float(x) / w)), max(0.0, min(1.0, float(y) / h))])
                    all_labels.add(lbl)
                    polys.append((lbl, norm_pts))
                parsed_entries.append((img_p, w, h, split, polys))
            except Exception:
                continue

        if parsed_entries and all_labels:
            cls_names = sorted(all_labels)
            cls_map = {c: i for i, c in enumerate(cls_names)}
            for idx, (img_p, _, _, split, polys) in enumerate(parsed_entries):
                stem = f"{split}_{idx:05d}_{img_p.stem[:40]}"
                shutil.copy2(img_p, ds_dir / "images" / split / f"{stem}{img_p.suffix.lower()}")
                lines = [
                    f"{cls_map[lbl]} " + " ".join(f"{v:.6f}" for v in pts)
                    for lbl, pts in polys
                ]
                (ds_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                summary["splits"][split] += 1
                summary["instances"][split] += len(lines)

            yaml_path = ds_dir / "yolo26_plantseg.yaml"
            y_lines = [
                f"path: {ds_dir.resolve()}",
                "train: images/train",
                f"val: {'images/val' if summary['splits']['val'] > 0 else 'images/train'}",
            ]
            if summary["splits"]["test"] > 0:
                y_lines.append("test: images/test")
            y_lines.append("names:")
            for cid, cname in enumerate(cls_names):
                y_lines.append(f"  {cid}: '{cname}'")
            yaml_path.write_text("\n".join(y_lines) + "\n", encoding="utf-8")
            summary["yaml_path"] = str(yaml_path)
            summary["classes_count"] = len(cls_names)
            summary["status"] = "READY"
            summary["note"] = "Converted genuine PlantSeg LabelMe polygons to YOLO26-Seg format."
            return summary

    if png_masks:
        summary["annotation_modality"] = "semantic_binary_or_multiclass_masks"
        for idx, mp in enumerate(png_masks):
            img_cands = [
                c for c in plantseg_root.rglob(f"{mp.stem}.*")
                if c != mp and c.suffix.lower() in IMG_EXTS
            ]
            if not img_cands:
                continue
            img_p = img_cands[0]
            dims = validate_image_file(img_p)
            if dims is None:
                continue
            mask_arr = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if mask_arr is None:
                continue
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            polys = mask_to_yolo_polygons(mask_arr, class_id=0)
            parts_l = {pt.lower() for pt in mp.parts} | {pt.lower() for pt in img_p.parts}
            split = "test" if "test" in parts_l else ("val" if "val" in parts_l else "train")
            stem = f"{split}_{idx:05d}_{img_p.stem[:40]}"
            shutil.copy2(img_p, ds_dir / "images" / split / f"{stem}{img_p.suffix.lower()}")
            lines = [f"{cid} " + " ".join(f"{v:.6f}" for v in pts) for cid, pts in polys]
            (ds_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            summary["splits"][split] += 1
            summary["instances"][split] += len(lines)

        if summary["splits"]["train"] > 0:
            yaml_path = ds_dir / "yolo26_plantseg.yaml"
            y_lines = [
                f"path: {ds_dir.resolve()}",
                "train: images/train",
                f"val: {'images/val' if summary['splits']['val'] > 0 else 'images/train'}",
            ]
            if summary["splits"]["test"] > 0:
                y_lines.append("test: images/test")
            y_lines.extend(["names:", "  0: 'pathological_lesion'"])
            yaml_path.write_text("\n".join(y_lines) + "\n", encoding="utf-8")
            summary["yaml_path"] = str(yaml_path)
            summary["classes_count"] = 1
            summary["status"] = "READY"
            summary["note"] = "Extracted contiguous lesion polygons from genuine PlantSeg masks for YOLO26-Seg."
            return summary

    return summary


def build_negative_control_index(
    pv_root: Optional[Path],
    pw_root: Optional[Path],
    out_dir: Path,
    max_per_source: int = 100,
) -> Dict[str, Any]:
    """
    Indexes healthy samples from PlantVillage (CC0) and PlantWild (CC BY-NC-ND 4.0)
    WITHOUT synthesizing bounding boxes, strictly for false-positive rate evaluation.
    """
    records: List[Dict[str, Any]] = []
    for src_name, src_root in (("PlantVillage", pv_root), ("PlantWild", pw_root)):
        if src_root is None or not src_root.exists():
            continue
        count = 0
        for d in sorted(src_root.iterdir()):
            if not d.is_dir():
                continue
            dl = d.name.lower()
            is_healthy = ("healthy" in dl) or (src_name == "PlantWild" and dl.endswith(" leaf"))
            if not is_healthy:
                continue
            for img_p in sorted(d.glob("*.*")):
                if img_p.suffix.lower() not in IMG_EXTS:
                    continue
                if validate_image_file(img_p) is None:
                    continue
                records.append({
                    "image_path": str(img_p.resolve()),
                    "dataset_source": src_name,
                    "class_label": d.name,
                    "license": DATASET_LICENSES[src_name]["license"],
                })
                count += 1
                if count >= max_per_source:
                    break
            if count >= max_per_source:
                break

    idx_path = out_dir / "healthy_negative_control_index.json"
    idx_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {"count": len(records), "index_path": str(idx_path)}


def run_preparation(search_roots: List[Path], out_dir: Path, seed: int = 42) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_tri_datasets(search_roots, extract_dir=out_dir / "_extracted")

    plantdoc_summary = prepare_plantdoc_yolo26(discovered["plantdoc_root"], out_dir, seed=seed)
    plantseg_summary = prepare_plantseg_lesion_yolo26(discovered["plantseg_root"], out_dir)
    neg_summary = build_negative_control_index(
        discovered["plantvillage_root"], discovered["plantwild_root"], out_dir
    )

    report = {
        "pipeline_version": "yolo26_tri_dataset_prep_v1",
        "discovered_paths": {k: (str(v) if v else None) for k, v in discovered.items()},
        "licenses_and_restrictions": DATASET_LICENSES,
        "plantdoc_specimen_detection": plantdoc_summary,
        "plantseg_lesion_segmentation": plantseg_summary,
        "healthy_negative_controls": neg_summary,
        "excluded_pseudo_datasets": [
            "cv/datasets/processed/yolo_lesions (HSV+LAB color thresholding — excluded)",
            "cv/datasets/processed/segmentation (HSV+LAB color thresholding — excluded)",
        ],
    }
    report_path = out_dir / "dataset_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-destructive YOLO26 Tri-Dataset Preparation")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Input roots to scan (defaults to /kaggle/input, cv/datasets/raw, data, .)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/yolo26_v1/datasets"),
        help="Versioned output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    roots = [Path(r) for r in args.search_root] if args.search_root else [
        Path("/kaggle/input"),
        Path("cv/datasets/raw"),
        Path("data"),
        Path("."),
    ]
    report = run_preparation(roots, args.out_dir, seed=args.seed)
    print(json.dumps({
        "plantdoc_status": report["plantdoc_specimen_detection"]["status"],
        "plantdoc_splits": report["plantdoc_specimen_detection"]["splits"],
        "plantseg_status": report["plantseg_lesion_segmentation"]["status"],
        "plantseg_splits": report["plantseg_lesion_segmentation"]["splits"],
        "healthy_controls": report["healthy_negative_controls"]["count"],
        "report_path": str(args.out_dir / "dataset_preparation_report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()

