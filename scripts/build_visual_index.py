#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"faiss import failed: {exc}")


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def rebuild_manifest_from_done(cache_dir: Path, manifest_path: Path) -> list[dict]:
    tokens_root = cache_dir / "tokens"
    if not tokens_root.exists():
        raise SystemExit(f"tokens dir not found: {tokens_root}")
    rows = []
    for sample_dir in sorted(tokens_root.iterdir()):
        done_flag = sample_dir / "done.json"
        if not done_flag.exists():
            continue
        rows.append(json.loads(done_flag.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no done.json found under {tokens_root}")
    rows.sort(key=lambda x: int(x.get("row_idx", 10**18)))
    with manifest_path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return rows


def iter_token_parts(cache_root: Path, entry: dict):
    for p in entry.get("parts", []):
        arr = np.load(cache_root / p["path"], mmap_mode="r")
        start = int(p["start"])
        end = int(p["end"])
        yield arr, start, end


def build_index_for_entry(cache_root: Path, entry: dict, out_path: Path, score_type: str = "dot") -> dict:
    d = int(entry.get("d_vis", 1024))
    base = faiss.IndexFlatIP(d)
    idx = faiss.IndexIDMap2(base)

    total = 0
    for arr, start, end in iter_token_parts(cache_root, entry):
        x = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        if score_type == "cosine":
            faiss.normalize_L2(x)
        ids = np.ascontiguousarray(np.arange(start, end, dtype=np.int64))
        idx.add_with_ids(x, ids)
        total += int(x.shape[0])

    faiss.write_index(idx, str(out_path))
    return {
        "sample_uid": entry["sample_uid"],
        "index_path": str(out_path),
        "score_type": score_type,
        "ntotal": int(idx.ntotal),
        "expected_total": int(entry.get("total_tokens", total)),
        "d_vis": d,
        "index_type": "IndexIDMap2(IndexFlatIP)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--index_dir", type=str, default="")
    ap.add_argument("--score_type", type=str, choices=["dot", "cosine"], default="dot")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"[warn] manifest not found, rebuilding from done.json: {manifest_path}")
        rows = rebuild_manifest_from_done(cache_dir, manifest_path)
    else:
        rows = load_manifest(manifest_path)

    index_dir = Path(args.index_dir) if args.index_dir else (cache_dir / "indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    meta_rows = []
    for i, entry in enumerate(rows):
        uid = entry["sample_uid"]
        out_path = index_dir / f"{uid}.faiss"
        if args.overwrite == 0 and out_path.exists():
            continue
        meta = build_index_for_entry(cache_dir, entry, out_path=out_path, score_type=args.score_type)
        meta_rows.append(meta)
        if (i + 1) % 10 == 0:
            print(f"built {i+1}/{len(rows)} indexes")

    meta_path = index_dir / "index_manifest.jsonl"
    existing = []
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                existing.append(json.loads(line))
    by_uid = {x["sample_uid"]: x for x in existing}
    for x in meta_rows:
        by_uid[x["sample_uid"]] = x

    with meta_path.open("w", encoding="utf-8") as f:
        for uid in sorted(by_uid.keys()):
            f.write(json.dumps(by_uid[uid], ensure_ascii=False) + "\n")

    summary = {
        "cache_dir": str(cache_dir),
        "index_dir": str(index_dir),
        "score_type": args.score_type,
        "num_built_this_run": len(meta_rows),
        "num_total_indexed": len(by_uid),
        "index_manifest": str(meta_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
