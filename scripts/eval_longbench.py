#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import extract_choice, gold_choice
from src.model import OpticalRAGModel


def choice_token_id(tokenizer, ch: str) -> int:
    for c in [f" {ch}", ch]:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(ch, add_special_tokens=False)
    return ids[0]


def apply_char_budget(context: str, budget_chars: int) -> str:
    # budget_chars <= 0 means "use full context"
    if budget_chars <= 0:
        return context
    return context[:budget_chars]


def baseline_predict(item, model, tokenizer, device, max_input_tokens, max_new_tokens, context_budget_chars):
    context = apply_char_budget(item["context"], context_budget_chars)
    prompt = (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{context}\n</text>\n\n"
        f"What is the correct answer to this question: {item['question']}\n"
        f"Choices:\n(A) {item['choice_A']}\n(B) {item['choice_B']}\n(C) {item['choice_C']}\n(D) {item['choice_D']}\n\n"
        "Format your response as follows: \"The correct answer is (insert answer here)\"."
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input_tokens:
        h = max_input_tokens // 2
        t = max_input_tokens - h
        ids = ids[:h] + ids[-t:]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def baseline_predict_choice(item, model, tokenizer, device, max_input_tokens, context_budget_chars):
    context = apply_char_budget(item["context"], context_budget_chars)
    prompt = (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{context}\n</text>\n\n"
        f"What is the correct answer to this question: {item['question']}\n"
        f"Choices:\n(A) {item['choice_A']}\n(B) {item['choice_B']}\n(C) {item['choice_C']}\n(D) {item['choice_D']}\n\n"
        "Output only one option letter in parentheses.\n"
        "The correct answer is ("
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input_tokens:
        h = max_input_tokens // 2
        t = max_input_tokens - h
        ids = ids[:h] + ids[-t:]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, return_dict=True)
    logits = out.logits[0, -1]
    option_ids = {c: choice_token_id(tokenizer, c) for c in ["A", "B", "C", "D"]}
    return max(option_ids.items(), key=lambda kv: float(logits[kv[1]].item()))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["baseline", "opticalrag"], required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--summary_json", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_input_tokens", type=int, default=3072)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--context_budget_chars", type=int, default=18000, help="Shared context budget fallback.")
    ap.add_argument(
        "--baseline_context_budget_chars",
        type=int,
        default=-1,
        help="Baseline-only context budget. >=0 takes priority over --context_budget_chars; <=0 means full context.",
    )
    ap.add_argument(
        "--opticalrag_context_budget_chars",
        type=int,
        default=0,
        help="OpticalRAG-only context budget. Default 0 means full context. Set >0 to force truncation.",
    )
    ap.add_argument("--decode_mode", type=str, choices=["generate", "choice_score"], default="choice_score")
    ap.add_argument("--longbench_local_path", type=str, default=str(ROOT / "data" / "LongBench-v2" / "train"))
    ap.add_argument("--model_path", type=str, default=str(ROOT / "external_models" / "Qwen2.5-7B-Instruct"))
    ap.add_argument("--deepseek_path", type=str, default=str(ROOT / "external_models" / "DeepSeek-OCR"))
    ap.add_argument("--opticalrag_context_sampling", type=str, choices=["head", "head_tail", "uniform"], default="head")
    ap.add_argument("--opticalrag_visual_window_chars", type=int, default=24000)
    ap.add_argument("--opticalrag_visual_num_windows", type=int, default=1)
    ap.add_argument("--opticalrag_visual_pages_per_window", type=int, default=4)
    ap.add_argument("--opticalrag_full_visual_encode", type=int, default=0)
    ap.add_argument("--opticalrag_full_visual_chunk_chars", type=int, default=24000)
    ap.add_argument("--opticalrag_full_visual_render_mode", type=str, choices=["text", "bytepack"], default="text")
    ap.add_argument("--opticalrag_retrieval_enable", type=int, default=0)
    ap.add_argument("--opticalrag_retrieval_topk", type=int, default=0)
    ap.add_argument("--opticalrag_retrieval_topk_factor", type=int, default=8)
    ap.add_argument("--opticalrag_retrieval_rerank_factor", type=int, default=4)
    ap.add_argument("--opticalrag_retrieval_backend", type=str, choices=["online", "offline_cache", "offline_index"], default="online")
    ap.add_argument(
        "--opticalrag_retrieval_score_type",
        type=str,
        choices=[
            "dot",
            "cosine",
            "l2",
            "maxsim",
            "maxsim_cosine",
            "dot_multi",
            "cosine_multi",
            "l2_multi",
            "dot_hybrid_max",
        ],
        default="dot",
    )
    ap.add_argument("--opticalrag_retrieval_score_alpha", type=float, default=0.25)
    ap.add_argument("--opticalrag_offline_cache_dir", type=str, default="")
    ap.add_argument("--opticalrag_offline_index_dir", type=str, default="")
    ap.add_argument("--opticalrag_offline_uid_prefix", type=str, default="longbench_train")
    ap.add_argument("--opticalrag_retrieval_preserve_order", type=int, choices=[0, 1], default=0)
    ap.add_argument("--opticalrag_use_downsample", type=int, choices=[0, 1], default=1)
    ap.add_argument("--opticalrag_downsample_stride", type=int, default=2)
    ap.add_argument("--opticalrag_prefilter_topk", type=int, default=0)
    ap.add_argument("--opticalrag_prefilter_mode", type=str, choices=["hard", "soft"], default="hard")
    ap.add_argument("--opticalrag_prefilter_soft_scale", type=float, default=1.0)
    ap.add_argument("--opticalrag_query_pooling", type=str, choices=["mean", "last", "max", "max_mean", "attn", "none"], default="mean")
    ap.add_argument("--opticalrag_compressor_mode", type=str, choices=["qdc", "mean_pool16", "rand_proj16"], default="qdc")
    ap.add_argument("--opticalrag_num_query_tokens", type=int, default=16)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:auto")
    args = ap.parse_args()

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        raise FileExistsError("Predictions already exist; choose a new output directory")

    if Path(args.longbench_local_path).exists():
        ds = load_from_disk(args.longbench_local_path)
    else:
        raise FileNotFoundError("Prepare the pinned dataset with scripts/download_assets.py first")
    items = list(ds)
    if args.max_samples > 0:
        items = items[: args.max_samples]

    baseline_budget = args.context_budget_chars if args.baseline_context_budget_chars < 0 else args.baseline_context_budget_chars
    opticalrag_budget = args.opticalrag_context_budget_chars

    if args.mode == "baseline":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32).to(device)
        model.eval()
    else:
        opticalrag = OpticalRAGModel(
            llm_path=args.model_path,
            deepseek_path=args.deepseek_path,
            device=args.device,
            num_query_tokens=args.opticalrag_num_query_tokens,
            context_sampling=args.opticalrag_context_sampling,
            query_pooling=args.opticalrag_query_pooling,
            compressor_mode=args.opticalrag_compressor_mode,
            pre_filter_topk=args.opticalrag_prefilter_topk,
            pre_filter_mode=args.opticalrag_prefilter_mode,
            pre_filter_soft_scale=args.opticalrag_prefilter_soft_scale,
            visual_window_chars=args.opticalrag_visual_window_chars,
            visual_num_windows=args.opticalrag_visual_num_windows,
            visual_pages_per_window=args.opticalrag_visual_pages_per_window,
            full_visual_encode=bool(args.opticalrag_full_visual_encode),
            full_visual_chunk_chars=args.opticalrag_full_visual_chunk_chars,
            full_visual_render_mode=args.opticalrag_full_visual_render_mode,
            retrieval_enable=bool(args.opticalrag_retrieval_enable),
            retrieval_topk=args.opticalrag_retrieval_topk,
            retrieval_topk_factor=args.opticalrag_retrieval_topk_factor,
            retrieval_rerank_factor=args.opticalrag_retrieval_rerank_factor,
            retrieval_backend=args.opticalrag_retrieval_backend,
            retrieval_score_type=args.opticalrag_retrieval_score_type,
            retrieval_score_alpha=args.opticalrag_retrieval_score_alpha,
            offline_cache_dir=args.opticalrag_offline_cache_dir,
            offline_index_dir=args.opticalrag_offline_index_dir,
            offline_uid_prefix=args.opticalrag_offline_uid_prefix,
            retrieval_preserve_order=bool(args.opticalrag_retrieval_preserve_order),
            use_depthwise_downsample=bool(args.opticalrag_use_downsample),
            downsample_stride=args.opticalrag_downsample_stride,
        )
        if args.ckpt:
            bundle = torch.load(args.ckpt, map_location=opticalrag.device_obj, weights_only=False)
            opticalrag.qcompress.load_state_dict(bundle["qcompress"], strict=True)
            opticalrag.bridge.load_state_dict(bundle["bridge"], strict=True)
        opticalrag.eval()

    stats = {"all":0,"hit":0,"easy":0,"easy_hit":0,"hard":0,"hard_hit":0,"short":0,"short_hit":0,"medium":0,"medium_hit":0,"long":0,"long_hit":0}

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(tqdm(items, desc=f"eval:{args.mode}")):
            if args.mode == "baseline":
                if args.decode_mode == "choice_score":
                    pred = baseline_predict_choice(item, model, tokenizer, device, args.max_input_tokens, baseline_budget)
                    response = f"The correct answer is ({pred})"
                else:
                    response = baseline_predict(item, model, tokenizer, device, args.max_input_tokens, args.max_new_tokens, baseline_budget)
                    pred = extract_choice(response)
            else:
                opticalrag_context = apply_char_budget(item["context"], opticalrag_budget)
                if args.decode_mode == "choice_score":
                    pred = opticalrag.predict_mcq_choice(
                        context=opticalrag_context,
                        question=item["question"],
                        choices=[item["choice_A"], item["choice_B"], item["choice_C"], item["choice_D"]],
                        sample_uid=i,
                    )
                    response = f"The correct answer is ({pred})"
                else:
                    response = opticalrag.predict_mcq(
                        context=opticalrag_context,
                        question=item["question"],
                        choices=[item["choice_A"], item["choice_B"], item["choice_C"], item["choice_D"]],
                        max_new_tokens=args.max_new_tokens,
                        sample_uid=i,
                    )
                    pred = extract_choice(response)

            gold = gold_choice(item["answer"])
            judge = int(pred == gold and pred != "")
            stats["all"] += 1
            stats["hit"] += judge

            if item["difficulty"] == "easy":
                stats["easy"] += 1
                stats["easy_hit"] += judge
            else:
                stats["hard"] += 1
                stats["hard_hit"] += judge

            if item["length"] == "short":
                stats["short"] += 1
                stats["short_hit"] += judge
            elif item["length"] == "medium":
                stats["medium"] += 1
                stats["medium_hit"] += judge
            else:
                stats["long"] += 1
                stats["long_hit"] += judge

            rec = dict(item)
            rec.update({"pred": pred, "gold": gold, "judge": bool(judge), "response": response})
            if args.mode == "opticalrag":
                rec["retrieval"] = dict(opticalrag.last_retrieval_stats)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    summary = {
        "mode": args.mode,
        "num_samples": stats["all"],
        "overall": round(100.0 * stats["hit"] / max(1, stats["all"]), 2),
        "easy": round(100.0 * stats["easy_hit"] / max(1, stats["easy"]), 2),
        "hard": round(100.0 * stats["hard_hit"] / max(1, stats["hard"]), 2),
        "short": round(100.0 * stats["short_hit"] / max(1, stats["short"]), 2),
        "medium": round(100.0 * stats["medium_hit"] / max(1, stats["medium"]), 2),
        "long": round(100.0 * stats["long_hit"] / max(1, stats["long"]), 2),
        "shared_context_budget_chars": args.context_budget_chars,
        "baseline_context_budget_chars": baseline_budget,
        "opticalrag_context_budget_chars": opticalrag_budget,
        "opticalrag_context_sampling": args.opticalrag_context_sampling,
        "opticalrag_query_pooling": args.opticalrag_query_pooling,
        "opticalrag_compressor_mode": args.opticalrag_compressor_mode,
        "opticalrag_num_query_tokens": args.opticalrag_num_query_tokens,
        "opticalrag_visual_window_chars": args.opticalrag_visual_window_chars,
        "opticalrag_visual_num_windows": args.opticalrag_visual_num_windows,
        "opticalrag_visual_pages_per_window": args.opticalrag_visual_pages_per_window,
        "opticalrag_full_visual_encode": int(args.opticalrag_full_visual_encode),
        "opticalrag_full_visual_chunk_chars": args.opticalrag_full_visual_chunk_chars,
        "opticalrag_full_visual_render_mode": args.opticalrag_full_visual_render_mode,
        "opticalrag_retrieval_enable": int(args.opticalrag_retrieval_enable),
        "opticalrag_retrieval_topk": args.opticalrag_retrieval_topk,
        "opticalrag_retrieval_topk_factor": args.opticalrag_retrieval_topk_factor,
        "opticalrag_retrieval_rerank_factor": args.opticalrag_retrieval_rerank_factor,
        "opticalrag_retrieval_backend": args.opticalrag_retrieval_backend,
        "opticalrag_retrieval_score_type": args.opticalrag_retrieval_score_type,
        "opticalrag_retrieval_score_alpha": args.opticalrag_retrieval_score_alpha,
        "opticalrag_offline_cache_dir": args.opticalrag_offline_cache_dir,
        "opticalrag_offline_index_dir": args.opticalrag_offline_index_dir,
        "opticalrag_offline_uid_prefix": args.opticalrag_offline_uid_prefix,
        "opticalrag_retrieval_preserve_order": int(args.opticalrag_retrieval_preserve_order),
        "opticalrag_use_downsample": int(args.opticalrag_use_downsample),
        "opticalrag_downsample_stride": args.opticalrag_downsample_stride,
        "opticalrag_prefilter_topk": args.opticalrag_prefilter_topk,
        "opticalrag_prefilter_mode": args.opticalrag_prefilter_mode,
        "opticalrag_prefilter_soft_scale": args.opticalrag_prefilter_soft_scale,
        "output_jsonl": str(output_jsonl),
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
