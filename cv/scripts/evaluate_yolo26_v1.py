#!/usr/bin/env python3
"""
evaluate_yolo26_v1.py
---------------------
Independent evaluation, architectural verification, latency benchmark, and
visual comparison harness comparing candidate YOLO26 models (specimen detect
and lesion seg/detect) against preserved production baselines (YOLO11n & YOLOv8n)
on identical held-out evaluation images.

Strict Scientific Rules:
  - Never fabricates metrics; missing checkpoints are reported as NOT_TRAINED.
  - Verifies true model architecture (yaml_file, end2end head, reg_max, C3k2 vs C2f)
    rather than trusting filenames.
  - Compares end2end=True (NMS-free) vs end2end=False (one-to-many + NMS).
  - Applies an explicit quantitative promotion gate before recommending switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_yolo_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """Extracts true Ultralytics checkpoint architecture, parameters, and SHA-256."""
    if not ckpt_path.exists():
        return {"status": "MISSING", "path": str(ckpt_path)}

    model = YOLO(str(ckpt_path))
    nn_mod = getattr(model, "model", None)
    yaml_info = getattr(nn_mod, "yaml", None) or {}
    yaml_file = str(yaml_info.get("yaml_file", "")).lower()

    backbone_ops = set()
    for layer in yaml_info.get("backbone", []) or []:
        if isinstance(layer, (list, tuple)) and len(layer) >= 3:
            backbone_ops.add(str(layer[2]))

    head = nn_mod.model[-1] if (nn_mod is not None and hasattr(nn_mod, "model") and len(nn_mod.model) > 0) else None
    has_o2m = hasattr(head, "one2many") if head is not None else False
    has_o2o = (hasattr(head, "one2one") or hasattr(head, "one2one_cv2")) if head is not None else False
    yaml_e2e = bool(yaml_info.get("end2end", False))
    head_e2e = bool(getattr(head, "end2end", False)) if head is not None else False
    mod_e2e = bool(getattr(nn_mod, "end2end", False)) if nn_mod is not None else False
    is_e2e = bool(head_e2e or mod_e2e or yaml_e2e or has_o2o)
    reg_max = getattr(head, "reg_max", None) if head is not None else None
    head_cls = type(head).__name__ if head is not None else "Unknown"
    is_yolo26 = bool("yolo26" in yaml_file and (has_o2o or head_e2e or mod_e2e or yaml_e2e))

    if is_yolo26:
        arch = f"YOLO26 ({head_cls})"
    elif "yolo11" in yaml_file or "C3k2" in backbone_ops:
        arch = f"YOLO11 ({head_cls})"
    elif "yolov8" in yaml_file or ("C2f" in backbone_ops and not is_e2e):
        arch = f"YOLOv8 ({head_cls})"
    else:
        arch = f"YOLO ({head_cls})"

    params = int(sum(p.numel() for p in nn_mod.parameters())) if nn_mod is not None else 0
    return {
        "status": "LOADED",
        "path": str(ckpt_path),
        "sha256": sha256_file(ckpt_path),
        "size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 3),
        "detected_architecture": arch,
        "is_genuine_yolo26": is_yolo26,
        "yaml_file": yaml_file or None,
        "head_class": head_cls,
        "end2end": is_e2e,
        "head_end2end": head_e2e,
        "one2many": has_o2m,
        "one2one": has_o2o,
        "reg_max": int(reg_max) if reg_max is not None else None,
        "num_classes": len(model.names),
        "params": params,
    }


def _xywh_norm_to_xyxy_px(xc: float, yc: float, bw: float, bh: float, w: int, h: int) -> List[float]:
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return [x1, y1, x2, y2]


def _box_iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def evaluate_on_yolo_split(
    ckpt_path: Path,
    img_dir: Path,
    lbl_dir: Path,
    yaml_names: Dict[int, str],
    imgsz: int = 640,
    conf: float = 0.20,
    iou_thresh: float = 0.50,
    end2end_override: Optional[bool] = None,
    match_class_by_name: bool = True,
) -> Dict[str, Any]:
    """
    Runs model predictions on an exact image/label split and computes Precision,
    Recall, F1, small-object recall (< 1% image area or < 32px), and latency.
    Matching by normalized lowercase class name guarantees fair comparison against
    historical checkpoints whose internal class ID ordering differs from canonical YAML.
    """
    if not ckpt_path.exists() or not img_dir.exists():
        return {"status": "SKIPPED_UNAVAILABLE"}

    model = YOLO(str(ckpt_path))
    model_names_lower = {int(k): str(v).strip().lower() for k, v in model.names.items()}
    yaml_names_lower = {int(k): str(v).strip().lower() for k, v in yaml_names.items()}

    img_paths = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        return {"status": "EMPTY_SPLIT"}

    tp_total = 0
    fp_total = 0
    fn_total = 0
    small_gt_total = 0
    small_tp_total = 0
    per_cls_stats: Dict[str, Dict[str, int]] = {}
    latencies_ms: List[float] = []

    def _cls_bucket(name: str) -> Dict[str, int]:
        if name not in per_cls_stats:
            per_cls_stats[name] = {"tp": 0, "fp": 0, "fn": 0}
        return per_cls_stats[name]

    # Warmup
    first_img = cv2.imread(str(img_paths[0]))
    if first_img is not None:
        kw = {"imgsz": imgsz, "conf": conf, "verbose": False}
        if end2end_override is not None:
            kw["end2end"] = end2end_override
        model.predict(first_img, **kw)

    for ip in img_paths:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        gt_boxes: List[Tuple[str, List[float], bool]] = []
        lp = lbl_dir / f"{ip.stem}.txt"
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cname = yaml_names_lower.get(cid, str(cid))
                if len(parts) == 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    xyxy = _xywh_norm_to_xyxy_px(xc, yc, bw, bh, iw, ih)
                else:
                    pts = list(map(float, parts[1:]))
                    xs = [pts[i] * iw for i in range(0, len(pts), 2)]
                    ys = [pts[i + 1] * ih for i in range(0, len(pts), 2)]
                    xyxy = [min(xs), min(ys), max(xs), max(ys)]
                box_w = xyxy[2] - xyxy[0]
                box_h = xyxy[3] - xyxy[1]
                is_small = ((box_w * box_h) / float(max(1, iw * ih)) < 0.01) or (max(box_w, box_h) < 32.0)
                gt_boxes.append((cname, xyxy, is_small))

        t0 = time.perf_counter()
        pred_kw = {"imgsz": imgsz, "conf": conf, "agnostic_nms": True, "verbose": False}
        if end2end_override is not None:
            pred_kw["end2end"] = end2end_override
        res = model.predict(img, **pred_kw)[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        preds: List[Tuple[float, str, List[float]]] = []
        if res.boxes is not None:
            for b in res.boxes:
                pcid = int(b.cls[0])
                pcname = model_names_lower.get(pcid, str(pcid))
                pconf = float(b.conf[0])
                pxyxy = b.xyxy[0].tolist()
                preds.append((pconf, pcname, pxyxy))
        preds.sort(key=lambda x: -x[0])

        matched_gt = set()
        for _, pcname, pxyxy in preds:
            best_iou = 0.0
            best_idx = -1
            for gi, (gcname, gxyxy, _) in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if match_class_by_name and pcname != gcname:
                    continue
                iou_v = _box_iou(pxyxy, gxyxy)
                if iou_v >= iou_thresh and iou_v > best_iou:
                    best_iou = iou_v
                    best_idx = gi
            if best_idx >= 0:
                matched_gt.add(best_idx)
                tp_total += 1
                _cls_bucket(pcname)["tp"] += 1
                if gt_boxes[best_idx][2]:
                    small_tp_total += 1
            else:
                fp_total += 1
                _cls_bucket(pcname)["fp"] += 1

        for gi, (gcname, _, is_small) in enumerate(gt_boxes):
            if is_small:
                small_gt_total += 1
            if gi not in matched_gt:
                fn_total += 1
                _cls_bucket(gcname)["fn"] += 1

    precision = tp_total / float(max(1, tp_total + fp_total))
    recall = tp_total / float(max(1, tp_total + fn_total))
    f1 = (2.0 * precision * recall) / float(max(1e-9, precision + recall))
    small_recall = (small_tp_total / float(small_gt_total)) if small_gt_total > 0 else None

    lat_arr = np.array(latencies_ms, dtype=np.float64) if latencies_ms else np.array([0.0])
    return {
        "status": "EVALUATED",
        "images_evaluated": len(img_paths),
        "imgsz": imgsz,
        "conf": conf,
        "end2end": end2end_override,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "small_gt_count": small_gt_total,
        "small_object_recall": round(small_recall, 4) if small_recall is not None else None,
        "mean_latency_ms": round(float(np.mean(lat_arr)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "per_class_f1": {
            k: round((2 * v["tp"]) / float(max(1, 2 * v["tp"] + v["fp"] + v["fn"])), 4)
            for k, v in sorted(per_cls_stats.items())
        },
    }


def evaluate_negative_controls(
    ckpt_path: Path,
    image_paths: List[Path],
    imgsz: int = 416,
    conf: float = 0.18,
    end2end_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """Measures spurious detections on healthy / non-lesion negative control images."""
    if not ckpt_path.exists() or not image_paths:
        return {"status": "SKIPPED_UNAVAILABLE"}

    model = YOLO(str(ckpt_path))
    fp_images = 0
    total_boxes = 0
    evaluated = 0
    for ip in image_paths:
        if not ip.exists():
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        evaluated += 1
        kw = {"imgsz": imgsz, "conf": conf, "verbose": False}
        if end2end_override is not None:
            kw["end2end"] = end2end_override
        res = model.predict(im, **kw)[0]
        cnt = len(res.boxes) if res.boxes is not None else 0
        if cnt > 0:
            fp_images += 1
            total_boxes += cnt

    return {
        "status": "EVALUATED",
        "images_evaluated": evaluated,
        "images_with_false_positives": fp_images,
        "false_positive_rate": round(fp_images / float(max(1, evaluated)), 4),
        "total_false_positive_boxes": total_boxes,
    }


def render_side_by_side_comparison(
    baseline_ckpt: Path,
    candidate_ckpt: Optional[Path],
    image_paths: List[Path],
    out_dir: Path,
    imgsz: int = 640,
    conf: float = 0.20,
) -> List[str]:
    """Saves side-by-side baseline vs YOLO26 prediction strips for human visual review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m_base = YOLO(str(baseline_ckpt)) if baseline_ckpt.exists() else None
    m_cand = YOLO(str(candidate_ckpt)) if (candidate_ckpt and candidate_ckpt.exists()) else None
    saved: List[str] = []

    def _draw(img_bgr: np.ndarray, model: Optional[YOLO], title: str) -> np.ndarray:
        canvas = img_bgr.copy()
        if model is not None:
            res = model.predict(canvas, imgsz=imgsz, conf=conf, agnostic_nms=True, verbose=False)[0]
            for b in (res.boxes or []):
                x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
                cname = res.names[int(b.cls[0])]
                cval = float(b.conf[0])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (82, 183, 136), 2)
                cv2.putText(canvas, f"{cname} {cval:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (82, 183, 136), 1)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (15, 23, 42), -1)
        cv2.putText(canvas, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 250, 240), 1)
        return canvas

    for ip in image_paths:
        if not ip.exists():
            continue
        im = cv2.imread(str(ip))
        if im is None:
            continue
        panels = [_draw(im, m_base, f"Baseline: {baseline_ckpt.name if m_base else 'None'}")]
        if m_cand is not None:
            panels.append(_draw(im, m_cand, f"Candidate YOLO26: {candidate_ckpt.name}"))
        combined = np.hstack(panels)
        dst = out_dir / f"compare_{ip.stem}.jpg"
        cv2.imwrite(str(dst), combined)
        saved.append(str(dst))
    return saved


def evaluate_all(
    out_dir: Path,
    baseline_specimen: Path = Path("cv/models/yolo_plantdoc_best.pt"),
    baseline_lesion: Path = Path("cv/models/yolov8n_lesions_best.pt"),
    candidate_specimen: Optional[Path] = Path("artifacts/yolo26_v1/checkpoints/yolo26_specimen_best.pt"),
    candidate_lesion: Optional[Path] = Path("artifacts/yolo26_v1/checkpoints/yolo26_lesion_best.pt"),
    plantdoc_test_dir: Optional[Path] = None,
    project_root: Path = Path("."),
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "checkpoints": {
            "baseline_specimen": inspect_yolo_checkpoint(baseline_specimen),
            "baseline_lesion": inspect_yolo_checkpoint(baseline_lesion),
            "candidate_yolo26_specimen": inspect_yolo_checkpoint(candidate_specimen) if candidate_specimen else {"status": "MISSING"},
            "candidate_yolo26_lesion": inspect_yolo_checkpoint(candidate_lesion) if candidate_lesion else {"status": "MISSING"},
        },
        "specimen_comparison": {},
        "negative_control_lesion_fp": {},
        "visual_comparisons": [],
        "promotion_decision": {
            "decision": "RETAIN_PRODUCTION_BASELINES",
            "reason": "Trained YOLO26 checkpoints not yet present or have not passed held-out gate.",
        },
    }

    if plantdoc_test_dir and (plantdoc_test_dir / "images" / "test").exists():
        import yaml
        yaml_p = plantdoc_test_dir / "yolo26_plantdoc.yaml"
        names = yaml.safe_load(yaml_p.read_text())["names"] if yaml_p.exists() else {}
        report["specimen_comparison"]["baseline_yolo11n"] = evaluate_on_yolo_split(
            baseline_specimen, plantdoc_test_dir / "images" / "test", plantdoc_test_dir / "labels" / "test", names, imgsz=416, conf=0.15
        )
        if candidate_specimen and candidate_specimen.exists():
            e2e_res = evaluate_on_yolo_split(
                candidate_specimen, plantdoc_test_dir / "images" / "test", plantdoc_test_dir / "labels" / "test", names, imgsz=640, conf=0.20, end2end_override=True
            )
            o2m_res = evaluate_on_yolo_split(
                candidate_specimen, plantdoc_test_dir / "images" / "test", plantdoc_test_dir / "labels" / "test", names, imgsz=640, conf=0.20, end2end_override=False
            )
            if yaml_p.exists():
                try:
                    val_out = YOLO(str(candidate_specimen)).val(data=str(yaml_p), split="test", imgsz=640, verbose=False)
                    if hasattr(val_out, "box"):
                        e2e_res["map50"] = round(float(val_out.box.map50), 4)
                        e2e_res["map50_95"] = round(float(val_out.box.map), 4)
                except Exception:
                    pass
            report["specimen_comparison"]["yolo26_e2e"] = e2e_res
            report["specimen_comparison"]["yolo26_one2many"] = o2m_res

    healthy_imgs = [
        project_root / "frontend/samples/apple__healthy__healthy.jpg",
        project_root / "frontend/samples/potato__healthy__healthy.jpg",
        project_root / "cv/test_images/Strawberry-Leaves.jpg",
    ]
    report["negative_control_lesion_fp"]["baseline_yolov8n_lesions"] = evaluate_negative_controls(
        baseline_lesion, healthy_imgs, imgsz=256, conf=0.14
    )
    if candidate_lesion and candidate_lesion.exists():
        report["negative_control_lesion_fp"]["candidate_yolo26_lesions"] = evaluate_negative_controls(
            candidate_lesion, healthy_imgs, imgsz=640, conf=0.20
        )

    vis_imgs = [
        project_root / "cv/test_images/images_7.jpg",
        project_root / "cv/test_images/tomato-badleaves.jpg",
        project_root / "cv/test_images/tomato_virus_06_zoom.jpg",
        project_root / "frontend/samples/tomato__fungal__early_blight.jpg",
    ]
    report["visual_comparisons"] = render_side_by_side_comparison(
        baseline_specimen, candidate_specimen, vis_imgs, out_dir / "visual_comparisons", imgsz=416, conf=0.15
    )

    cand_spec_meta = report["checkpoints"]["candidate_yolo26_specimen"]
    base_spec_eval = report["specimen_comparison"].get("baseline_yolo11n", {})
    cand_spec_eval = report["specimen_comparison"].get("yolo26_e2e", {})
    if (
        cand_spec_meta.get("is_genuine_yolo26")
        and cand_spec_eval.get("status") == "EVALUATED"
        and base_spec_eval.get("status") == "EVALUATED"
        and cand_spec_eval["f1_score"] >= base_spec_eval["f1_score"]
        and cand_spec_eval["mean_latency_ms"] <= max(250.0, base_spec_eval["mean_latency_ms"] * 2.0)
    ):
        report["promotion_decision"] = {
            "decision": "ELIGIBLE_FOR_PROMOTION_REVIEW",
            "reason": "Genuine YOLO26 candidate met or exceeded YOLO11n F1 without unacceptable latency regression.",
        }

    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 candidates vs existing YOLO11n/YOLOv8n baselines")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/yolo26_v1"))
    parser.add_argument("--candidate-specimen", type=Path, default=Path("artifacts/yolo26_v1/checkpoints/yolo26_specimen_best.pt"))
    parser.add_argument("--candidate-lesion", type=Path, default=Path("artifacts/yolo26_v1/checkpoints/yolo26_lesion_best.pt"))
    parser.add_argument("--plantdoc-dir", type=Path, default=Path("artifacts/yolo26_v1/datasets/plantdoc_yolo26"))
    args = parser.parse_args()

    rep = evaluate_all(
        out_dir=args.out_dir,
        candidate_specimen=args.candidate_specimen,
        candidate_lesion=args.candidate_lesion,
        plantdoc_test_dir=args.plantdoc_dir if args.plantdoc_dir.exists() else None,
    )
    print(json.dumps(rep["checkpoints"], indent=2))
    print("Promotion decision:", rep["promotion_decision"])


if __name__ == "__main__":
    main()
