from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

from safetensors.torch import load_file


@lru_cache(maxsize=2)
def load_deepencoder_module(model_path: str) -> object:
    module_path = Path(model_path) / "deepencoder.py"
    spec = importlib.util.spec_from_file_location("opticalrag_deepencoder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def load_full_state_dict(model_path: str) -> dict:
    root = Path(model_path)
    if not root.is_dir():
        return load_file(str(root))

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        state_dict = {}
        loaded = set()
        for shard_name in index["weight_map"].values():
            if shard_name in loaded:
                continue
            state_dict.update(load_file(str(root / shard_name)))
            loaded.add(shard_name)
        return state_dict

    onefile = root / "model-00001-of-000001.safetensors"
    if onefile.exists():
        return load_file(str(onefile))

    raise FileNotFoundError(f"No safetensors found under {root}")
