#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import resolve_runtime_device
from src.identifiers import cache_sample_uid
from src.encoder import DeepEncoder, preprocess_image
from src.render import render_text_to_pages, render_text_to_pages_bytepack
from src.render_density_policy import RenderDensityPolicy


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    timeout_sec = int(os.environ.get("OPTICALRAG_DIST_TIMEOUT_SEC", "14400"))
    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_sec))
    return local_rank, global_rank, ws


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def load_samples(args) -> list[dict]:
    rows: list[dict] = []
    p = Path(args.longbench_local_path)
    ds = load_from_disk(str(p))
    for i, x in enumerate(ds):
        rows.append({"sample_uid": f"{args.uid_prefix}_{i:06d}",
                     "context": x.get("context", ""), "source": "longbench_v2", "idx": i})
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    return rows


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    t = (text or "").strip()
    if not t:
        return [""]
    if chunk_chars <= 0:
        return [t]
    return [t[i : i + chunk_chars] for i in range(0, len(t), chunk_chars)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longbench_local_path", type=str, default=str(ROOT / "data" / "LongBench-v2" / "train"))
    ap.add_argument("--uid_prefix", type=str, default="longbench_train")
    ap.add_argument("--max_samples", type=int, default=0)

    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--deepseek_path", type=str, default=str(ROOT / "external_models" / "DeepSeek-OCR"))
    ap.add_argument("--device", type=str, default="cuda:auto")

    ap.add_argument("--chunk_chars", type=int, default=24000)
    ap.add_argument("--page_batch", type=int, default=4)
    ap.add_argument("--render_mode", type=str, choices=["text", "bytepack"], default="text")
    ap.add_argument("--page_size", type=int, default=640)
    ap.add_argument("--font_size", type=int, default=-1)
    ap.add_argument("--margin", type=int, default=-1)
    ap.add_argument("--line_height_scale", type=float, default=-1.0)
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    ap.add_argument("--deduplicate_contexts", type=int, choices=[0, 1], default=1)
    args = ap.parse_args()

    local_rank, _, _ = setup_distributed()
    runtime_device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available() and is_distributed()
        else resolve_runtime_device(args.device)
    )

    cache_dir = Path(args.cache_dir)
    tokens_root = cache_dir / "tokens"
    tmp_dir = cache_dir / ".tmp_manifest_parts"
    if is_main():
        tokens_root.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()

    policy = RenderDensityPolicy.from_env()
    encoder = DeepEncoder(checkpoint=args.deepseek_path).to(runtime_device)
    encoder.eval()

    rows = load_samples(args)
    rank_manifest = tmp_dir / f"rank{rank():02d}.jsonl"
    started = time.time()
    done_local = 0
    # Only reuse documents encoded in this invocation, under identical settings.
    # Old cache entries retain the existing skip/resume behavior.
    encoded_contexts: dict[str, dict] = {}
    with rank_manifest.open("w", encoding="utf-8") as mf:
        for i in range(rank(), len(rows), world_size()):
            row = rows[i]
            uid = row["sample_uid"]
            context = (row.get("context") or "").strip()
            sample_dir = tokens_root / uid
            sample_dir.mkdir(parents=True, exist_ok=True)

            done_flag = sample_dir / "done.json"
            if args.overwrite == 0 and done_flag.exists():
                rec = json.loads(done_flag.read_text(encoding="utf-8"))
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
            if args.deduplicate_contexts and context_hash in encoded_contexts:
                original = encoded_contexts[context_hash]
                rec = dict(original)
                rec.update(
                    sample_uid=uid,
                    source=row.get("source", ""),
                    row_idx=int(row.get("idx", i)),
                    reused_from=original["sample_uid"],
                )
                done_flag.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_local += 1
                if done_local % 25 == 0:
                    print(f"[rank {rank():02d}] completed {done_local} samples; reused identical context", flush=True)
                continue

            parts = []
            total_tokens = 0
            part_id = 0
            density = policy.choose(len(context))
            if args.font_size > 0:
                density = density.__class__(args.font_size, density.margin, density.line_height_scale)
            if args.margin > 0:
                density = density.__class__(density.font_size, args.margin, density.line_height_scale)
            if args.line_height_scale > 0:
                density = density.__class__(density.font_size, density.margin, args.line_height_scale)

            chunks = split_chunks(context, args.chunk_chars)
            for chunk in chunks:
                if args.render_mode == "bytepack":
                    pages = render_text_to_pages_bytepack(
                        chunk,
                        page_size=args.page_size,
                        max_pages=0,
                        max_chars=0,
                    )
                else:
                    pages = render_text_to_pages(
                        chunk,
                        page_size=args.page_size,
                        font_size=density.font_size,
                        margin=density.margin,
                        line_height_scale=density.line_height_scale,
                        max_pages=0,
                        max_chars=0,
                    )
                for p in range(0, len(pages), max(1, args.page_batch)):
                    ims = [
                        preprocess_image(im, size=args.page_size, device=runtime_device, dtype=torch.float32)
                        for im in pages[p : p + max(1, args.page_batch)]
                    ]
                    x = torch.cat(ims, dim=0)
                    with torch.no_grad():
                        _, comp = encoder(x)
                    tok = comp.reshape(-1, comp.shape[-1]).detach().float().cpu().numpy().astype(np.float16)

                    rel = Path("tokens") / uid / f"part_{part_id:06d}.npy"
                    out_path = cache_dir / rel
                    np.save(out_path, tok)
                    start = total_tokens
                    end = total_tokens + int(tok.shape[0])
                    parts.append({"path": str(rel), "start": start, "end": end, "num_tokens": int(tok.shape[0])})
                    total_tokens = end
                    part_id += 1

            rec = {
                "sample_uid": uid,
                "source": row.get("source", ""),
                "row_idx": int(row.get("idx", i)),
                "context_chars": len(context),
                "context_sha256": context_hash,
                "encoder_device": str(runtime_device),
                "d_vis": 1024,
                "total_tokens": int(total_tokens),
                "num_parts": len(parts),
                "parts": parts,
            }
            encoded_contexts[context_hash] = rec
            done_flag.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done_local += 1
            if done_local % 10 == 0:
                elapsed = max(time.time() - started, 1e-6)
                rate = done_local / elapsed
                remain = max(0, ((len(rows) + world_size() - 1) // world_size()) - done_local)
                eta_min = remain / max(rate, 1e-6) / 60.0
                print(
                    f"[rank {rank():02d}] built {done_local}/{((len(rows) + world_size() - 1) // world_size())} "
                    f"samples | {rate:.3f} samples/s | ETA {eta_min:.1f} min",
                    flush=True,
                )

    if is_distributed():
        dist.barrier()

    if is_main():
        entries = []
        for sample_dir in sorted(tokens_root.iterdir()):
            done_flag = sample_dir / "done.json"
            if not done_flag.exists():
                continue
            entries.append(json.loads(done_flag.read_text(encoding="utf-8")))
        if not entries:
            for p in sorted(tmp_dir.glob("rank*.jsonl")):
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        entries.append(json.loads(line))
        entries.sort(key=lambda x: x["row_idx"])

        manifest = cache_dir / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for x in entries:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        stats = {
            "num_samples": len(entries),
            "total_tokens": int(sum(int(x.get("total_tokens", 0)) for x in entries)),
            "avg_tokens_per_sample": float(sum(int(x.get("total_tokens", 0)) for x in entries) / max(1, len(entries))),
            "render_mode": args.render_mode,
            "page_size": int(args.page_size),
            "chunk_chars": int(args.chunk_chars),
            "cache_dir": str(cache_dir),
            "manifest": str(manifest),
            "deduplicate_contexts": bool(args.deduplicate_contexts),
        }
        (cache_dir / "cache_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
