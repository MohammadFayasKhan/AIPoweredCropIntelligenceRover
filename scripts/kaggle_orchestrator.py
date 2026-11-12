#!/usr/bin/env python3
"""
Lightweight Antigravity <-> Kaggle GPU Orchestrator CLI.
Reads credentials strictly from ~/.kaggle/kaggle.json (never from repo files)
and supports validating credentials, pushing notebooks with external datasets
attached (GPU + Internet), polling status, and downloading outputs locally.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

API_BASE = "https://api.kaggle.com/v1"


def load_auth_headers() -> tuple[str, Dict[str, str]]:
    """Loads HTTP Basic Auth credentials securely from ~/.kaggle/kaggle.json."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        raise FileNotFoundError(f"Kaggle credentials not found at {cred_path}")
    creds = json.loads(cred_path.read_text(encoding="utf-8"))
    username = str(creds.get("username", "")).strip()
    key = str(creds.get("key", "")).strip()
    if not username or not key:
        raise ValueError(f"Incomplete credentials in {cred_path} (need 'username' and 'key')")
    token = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    return username, {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "CvProject-Kaggle-Orchestrator/1.0",
    }


def api_request(
    method: str,
    path: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} on {method} {path}: {err_body}") from err


def split_kernel_ref(ref: str, default_owner: str) -> tuple[str, str]:
    if "/" in ref:
        owner, slug = ref.split("/", 1)
        return owner.strip(), slug.strip()
    return default_owner, ref.strip()


def cmd_validate(args: argparse.Namespace) -> int:
    username, headers = load_auth_headers()
    blob_check = api_request(
        "POST",
        "/blobs/upload",
        headers,
        payload={
            "type": "dataset",
            "name": "antigravity-kaggle-credential-check.txt",
            "contentLength": 0,
            "contentType": "application/zip",
            "lastModifiedEpochSeconds": int(time.time()),
        },
    )
    print(f"Auth OK for user: {username} (upload handshake keys={list(blob_check.keys())})")
    if args.dataset:
        owner, dslug = split_kernel_ref(args.dataset, username)
        ds = api_request("GET", f"/datasets/view/{urllib.parse.quote(owner)}/{urllib.parse.quote(dslug)}", headers)
        size_gb = round(float(ds.get("totalBytesNullable") or 0) / (1024 ** 3), 2)
        print(f"Dataset OK: {owner}/{dslug} (size={size_gb} GB)")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    username, headers = load_auth_headers()
    nb_path = Path(args.notebook)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    nb_text = nb_path.read_text(encoding="utf-8")
    owner, kslug = split_kernel_ref(args.slug, username)
    title = args.title or kslug.replace("-", " ").title()
    payload = {
        "slug": f"{owner}/{kslug}",
        "newTitle": title,
        "text": nb_text,
        "language": "python",
        "kernelType": "notebook",
        "datasetDataSources": args.dataset or [],
        "kernelDataSources": [],
        "competitionDataSources": [],
        "modelDataSources": [],
        "categoryIds": [],
        "isPrivate": not args.public,
        "enableGpu": bool(args.gpu),
        "enableTpu": False,
        "enableInternet": bool(args.internet),
        "dockerImagePinningType": "latest",
    }
    if args.gpu and args.machine_shape:
        payload["machineShape"] = args.machine_shape
    resp = api_request("POST", "/kernels/push", headers, payload=payload)
    if resp.get("error"):
        print("Push failed:", resp["error"], file=sys.stderr)
        return 1
    bad = [
        *(resp.get("invalidDatasetSources") or []),
        *(resp.get("invalidKernelSources") or []),
        *(resp.get("invalidCompetitionSources") or []),
        *(resp.get("invalidModelSources") or []),
    ]
    if bad:
        print("Invalid data sources:", bad, file=sys.stderr)
        return 1
    print(json.dumps({"ref": resp.get("ref", f"{owner}/{kslug}"), "version": resp.get("versionNumber"), "url": resp.get("url")}, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    username, headers = load_auth_headers()
    owner, kslug = split_kernel_ref(args.kernel, username)
    st = api_request("GET", "/kernels/status", headers, params={"userName": owner, "kernelSlug": kslug})
    print(json.dumps({"kernel": f"{owner}/{kslug}", **st}, indent=2))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    username, headers = load_auth_headers()
    owner, kslug = split_kernel_ref(args.kernel, username)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    last_log = ""
    while True:
        params = {"userName": owner, "kernelSlug": kslug}
        if page_token:
            params["pageToken"] = page_token
        resp = api_request("GET", "/kernels/output", headers, params=params)
        if resp.get("log"):
            last_log = resp["log"]
        all_files.extend(resp.get("files") or [])
        if resp.get("hasNextPageToken") and resp.get("nextPageToken"):
            page_token = resp["nextPageToken"]
        else:
            break

    if args.save_log and last_log:
        (out_dir / "kernel_output.log").write_text(last_log, encoding="utf-8")

    downloaded = []
    for item in all_files:
        fname = item.get("fileName") or ""
        url = item.get("url") or ""
        if not fname or not url:
            continue
        if args.pattern and args.pattern not in fname:
            continue
        target = (out_dir / fname).resolve()
        if not str(target).startswith(str(out_dir.resolve())):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=180) as r:
            target.write_bytes(r.read())
        downloaded.append(str(target.relative_to(out_dir)))

    print(json.dumps({"kernel": f"{owner}/{kslug}", "matched_files": len(downloaded), "downloaded": downloaded}, indent=2))
    return 0


def cmd_upload_dataset(args: argparse.Namespace) -> int:
    import io
    import zipfile

    username, headers = load_auth_headers()
    owner, dslug = split_kernel_ref(args.slug, username)
    src_dir = Path(args.dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {src_dir}")

    buf = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                zf.write(p, str(p.relative_to(src_dir)))
                file_count += 1
    payload_bytes = buf.getvalue()

    blob = api_request(
        "POST",
        "/blobs/upload",
        headers,
        payload={
            "type": "dataset",
            "name": f"{dslug}.zip",
            "contentLength": len(payload_bytes),
            "contentType": "application/zip",
            "lastModifiedEpochSeconds": int(time.time()),
        },
    )
    create_url = blob.get("createUrl")
    token = blob.get("token")
    if not create_url or not token:
        raise RuntimeError(f"Unexpected /blobs/upload response: {blob}")

    put_req = urllib.request.Request(
        create_url,
        data=payload_bytes,
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(payload_bytes)),
        },
        method="PUT",
    )
    with urllib.request.urlopen(put_req, timeout=300) as resp:
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"GCS blob PUT failed with HTTP {resp.status}")

    exists = False
    try:
        api_request("GET", f"/datasets/view/{urllib.parse.quote(owner)}/{urllib.parse.quote(dslug)}", headers)
        exists = True
    except Exception:
        exists = False

    if exists:
        resp = api_request(
            "POST",
            f"/datasets/create/version/{urllib.parse.quote(owner)}/{urllib.parse.quote(dslug)}",
            headers,
            payload={
                "versionNotes": args.notes or "Updated eval assets from Antigravity orchestrator",
                "files": [{"token": token, "description": "Evaluation assets bundle"}],
                "convertToCsv": False,
                "deleteOldVersions": False,
                "categoryIds": [],
            },
        )
    else:
        resp = api_request(
            "POST",
            "/datasets/create/new",
            headers,
            payload={
                "title": args.title or dslug.replace("-", " ").title(),
                "slug": dslug,
                "ownerSlug": owner,
                "licenseName": "CC0-1.0",
                "subtitle": "Private Antigravity eval checkpoints and test images",
                "description": args.notes or "Evaluation checkpoints and test assets for YOLO26 Lesion Pipeline",
                "files": [{"token": token, "description": "Evaluation assets bundle"}],
                "isPrivate": True,
                "convertToCsv": False,
                "categoryIds": [],
            },
        )

    if resp.get("error"):
        raise RuntimeError(f"Dataset create/version failed: {resp}")

    # Poll briefly until Kaggle marks the dataset ready
    status = "unknown"
    for _ in range(30):
        try:
            st = api_request("GET", f"/datasets/status/{urllib.parse.quote(owner)}/{urllib.parse.quote(dslug)}", headers)
            status = str(st.get("status", "unknown"))
        except Exception:
            ds = api_request("GET", f"/datasets/view/{urllib.parse.quote(owner)}/{urllib.parse.quote(dslug)}", headers)
            if int(ds.get("totalBytesNullable") or 0) > 0:
                status = "ready"
                break
        if status.lower() in ("ready", "error"):
            break
        time.sleep(2)

    print(json.dumps({
        "dataset": f"{owner}/{dslug}",
        "action": "new_version" if exists else "create_new",
        "files_zipped": file_count,
        "zip_mb": round(len(payload_bytes) / 1024**2, 2),
        "status": status,
    }, indent=2))
    return 0 if status.lower() == "ready" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Antigravity Kaggle GPU Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate ~/.kaggle/kaggle.json credentials & dataset access")
    p_val.add_argument("--dataset", default="mohammadfayaskhan/smart-plant-codebase")
    p_val.set_defaults(func=cmd_validate)

    p_push = sub.add_parser("push", help="Push notebook (.ipynb) to Kaggle remote kernel")
    p_push.add_argument("notebook")
    p_push.add_argument("--slug", required=True)
    p_push.add_argument("--title", default=None)
    p_push.add_argument("--dataset", action="append", default=[])
    p_push.add_argument("--gpu", action="store_true", default=False)
    p_push.add_argument("--machine-shape", default="NvidiaTeslaT4")
    p_push.add_argument("--internet", action="store_true", default=False)
    p_push.add_argument("--public", action="store_true", default=False)
    p_push.set_defaults(func=cmd_push)

    p_st = sub.add_parser("status", help="Check Kaggle kernel status")
    p_st.add_argument("kernel")
    p_st.set_defaults(func=cmd_status)

    p_pull = sub.add_parser("pull", help="Download kernel outputs & log")
    p_pull.add_argument("kernel")
    p_pull.add_argument("--out", default="artifacts/kaggle_outputs")
    p_pull.add_argument("--pattern", default="")
    p_pull.add_argument("--save-log", action="store_true", default=True)
    p_pull.set_defaults(func=cmd_pull)

    p_up = sub.add_parser("upload-dataset", help="Zip a local folder and upload/update a private Kaggle dataset")
    p_up.add_argument("dir")
    p_up.add_argument("--slug", required=True)
    p_up.add_argument("--title", default=None)
    p_up.add_argument("--notes", default=None)
    p_up.set_defaults(func=cmd_upload_dataset)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

