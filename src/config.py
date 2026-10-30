from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.getenv("OPTICALRAG_MODEL_ROOT", PACKAGE_ROOT / "external_models"))
DATA_ROOT = Path(os.getenv("OPTICALRAG_DATA_ROOT", PACKAGE_ROOT / "data"))

_DEEPSEEK_OCR_CANDIDATES = (
    MODEL_ROOT / "DeepSeek-OCR",
)

_QWEN_7B_CANDIDATES = (
    MODEL_ROOT / "Qwen" / "Qwen2___5-7B-Instruct",
    MODEL_ROOT / "Qwen" / "Qwen2.5-7B-Instruct",
)


def _resolve_existing_path(explicit: str | Path | None, candidates: tuple[Path, ...], label: str) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"{label} not found: {p}")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve {label}; tried: {', '.join(map(str, candidates))}")


def resolve_deepseek_ocr_path(path: str | Path | None = None) -> Path:
    return _resolve_existing_path(path, _DEEPSEEK_OCR_CANDIDATES, "DeepSeek-OCR model")


def resolve_llm_path(path: str | Path | None = None) -> Path:
    return _resolve_existing_path(path, _QWEN_7B_CANDIDATES, "Qwen2.5-7B-Instruct")


DEFAULT_DEEPSEEK_OCR_PATH = None
DEFAULT_LLM_PATH = None


@lru_cache(maxsize=2)
def get_llm_hidden_size(path: str | Path | None = None) -> int:
    p = resolve_llm_path(path)
    with (p / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["hidden_size"])


def resolve_runtime_device(device_name: str = "cuda:auto") -> torch.device:
    if device_name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_name in {"cuda:auto", "cuda"}:
        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes > best_free:
                best_idx = idx
                best_free = free_bytes
        return torch.device(f"cuda:{best_idx}")
    return torch.device(device_name)
