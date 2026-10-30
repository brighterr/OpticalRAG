from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .query_visual_scoring import QueryVisualScorer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class OfflineRetrievalOutput:
    tokens: torch.Tensor
    indices: torch.Tensor
    scores: torch.Tensor
    total_tokens: int


class VisualCacheStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.jsonl"
        self._entries = self._load_entries()

    def _load_entries(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not self.manifest_path.exists():
            return out
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out[row["sample_uid"]] = row
        return out

    def has(self, sample_uid: str) -> bool:
        return sample_uid in self._entries

    def get(self, sample_uid: str) -> dict[str, Any]:
        return self._entries[sample_uid]

    def total_tokens(self, sample_uid: str) -> int:
        return int(self.get(sample_uid)["total_tokens"])

    def iter_parts(self, sample_uid: str) -> Iterable[tuple[np.ndarray, int, int]]:
        entry = self.get(sample_uid)
        for p in entry.get("parts", []):
            rel = p["path"]
            start = int(p["start"])
            end = int(p["end"])
            arr = np.load(self.root / rel, mmap_mode="r")
            yield arr, start, end

    def gather_by_indices(self, sample_uid: str, indices: np.ndarray) -> np.ndarray:
        if indices.ndim != 1:
            indices = indices.reshape(-1)
        if len(indices) == 0:
            return np.zeros((0, 1024), dtype=np.float32)

        order = np.argsort(indices)
        sorted_idx = indices[order]
        out = np.zeros((len(indices), 1024), dtype=np.float32)

        cursor = 0
        for arr, start, end in self.iter_parts(sample_uid):
            while cursor < len(sorted_idx) and sorted_idx[cursor] < start:
                cursor += 1
            if cursor >= len(sorted_idx):
                break
            j = cursor
            while j < len(sorted_idx) and sorted_idx[j] < end:
                j += 1
            if j == cursor:
                continue
            local = sorted_idx[cursor:j] - start
            block = np.asarray(arr[local], dtype=np.float32)
            orig_pos = order[cursor:j]
            out[orig_pos] = block
            cursor = j
        return out


class OfflineIndexStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] = {}

    def index_path(self, sample_uid: str) -> Path:
        return self.root / f"{sample_uid}.faiss"

    def has(self, sample_uid: str) -> bool:
        return self.index_path(sample_uid).exists()

    def load(self, sample_uid: str):
        if sample_uid not in self._cache:
            if faiss is None:
                raise RuntimeError("faiss is not available")
            # Evaluation visits many large indexes; retain only the current one.
            # Evict before loading to avoid accumulating the full dataset in RAM.
            self._cache.clear()
            self._cache[sample_uid] = faiss.read_index(str(self.index_path(sample_uid)))
        return self._cache[sample_uid]


class OfflineVisualRetriever:
    """
    Retrieval backend on top of offline visual-token cache + optional FAISS index.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        index_dir: str | Path | None = None,
        score_type: str = "dot",
        score_alpha: float = 0.25,
        backend: str = "offline_cache",
        preserve_original_order: bool = False,
    ):
        self.backend = backend
        self.cache = VisualCacheStore(cache_dir)
        self.index = OfflineIndexStore(index_dir or (Path(cache_dir) / "indexes"))
        self.scorer = QueryVisualScorer(score_type=score_type, score_alpha=score_alpha)
        self.score_type = score_type
        self.preserve_original_order = bool(preserve_original_order)
        st = str(score_type).strip().lower()
        self.recall_score_type = "cosine" if st in {"cosine", "maxsim_cosine", "cosine_multi"} else "dot"

    def _maybe_reorder_by_index(self, ids: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.preserve_original_order or len(ids) <= 1:
            return ids, scores
        order = np.argsort(ids, kind="stable")
        return ids[order], scores[order]

    def _search_faiss(self, sample_uid: str, query_global: torch.Tensor, topk: int) -> tuple[np.ndarray, np.ndarray]:
        if faiss is None:
            raise RuntimeError("faiss is not available")
        if self.recall_score_type not in {"dot", "cosine"}:
            raise RuntimeError(f"faiss recall only supports dot/cosine, got: {self.recall_score_type}")
        idx = self.index.load(sample_uid)
        q = query_global[0, 0].detach().float().cpu().numpy().reshape(1, -1).astype(np.float32)
        if self.recall_score_type == "cosine":
            faiss.normalize_L2(q)
        scores, ids = idx.search(q, int(topk))
        ids = ids[0]
        scores = scores[0]
        keep = ids >= 0
        ids = ids[keep].astype(np.int64)
        scores = scores[keep].astype(np.float32)
        return self._maybe_reorder_by_index(ids, scores)

    def _search_cache(self, sample_uid: str, query_repr: torch.Tensor, topk: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
        q = query_repr.to(device=device, dtype=torch.float32)
        best_scores = torch.empty((0,), device=device, dtype=torch.float32)
        best_indices = torch.empty((0,), device=device, dtype=torch.long)

        for arr, start, end in self.cache.iter_parts(sample_uid):
            v = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device=device)
            s = self.scorer.score(v.unsqueeze(0), q).squeeze(0)
            part_idx = torch.arange(start, end, device=device, dtype=torch.long)

            cand_scores = torch.cat([best_scores, s], dim=0)
            cand_indices = torch.cat([best_indices, part_idx], dim=0)
            k = min(int(topk), int(cand_scores.numel()))
            tk = cand_scores.topk(k=k, dim=0, largest=True, sorted=False)
            best_scores = tk.values
            best_indices = cand_indices[tk.indices]

        ids, scores = self._maybe_reorder_by_index(
            best_indices.detach().cpu().numpy().astype(np.int64),
            best_scores.detach().cpu().numpy().astype(np.float32),
        )
        return ids, scores

    def retrieve(
        self,
        sample_uid: str,
        query_global: torch.Tensor,
        topk: int,
        device: torch.device,
        query_repr: torch.Tensor | None = None,
        recall_topk_factor: int = 4,
    ) -> OfflineRetrievalOutput:
        if not self.cache.has(sample_uid):
            raise KeyError(f"Sample not found in offline cache: {sample_uid}")
        total = self.cache.total_tokens(sample_uid)
        k = max(1, min(int(topk), int(total)))
        q_repr = query_global if query_repr is None else query_repr

        st = str(self.score_type).strip().lower()
        multi_rerank_types = {"maxsim", "maxsim_cosine", "dot_multi", "cosine_multi", "l2_multi", "dot_hybrid_max"}
        use_index = self.backend == "offline_index" and self.index.has(sample_uid) and st in {"dot", "cosine", *multi_rerank_types}
        if self.backend == "offline_index" and not self.index.has(sample_uid):
            raise FileNotFoundError(f"Requested FAISS index missing: {self.index.index_path(sample_uid)}")
        self.last_backend = "faiss" if use_index else "cache_scan"
        if use_index:
            recall_k = k
            if st in multi_rerank_types:
                recall_k = min(int(total), max(int(k), int(k) * max(1, int(recall_topk_factor))))
            ids, scores = self._search_faiss(sample_uid, query_global, topk=recall_k)
            if st in multi_rerank_types and len(ids) > 0:
                cand_tok = self.cache.gather_by_indices(sample_uid, ids)
                cand = torch.from_numpy(cand_tok).to(device=device, dtype=torch.float32).unsqueeze(0)
                rerank_scores = self.scorer.score(cand, q_repr.to(device=device, dtype=torch.float32)).squeeze(0)
                final_k = min(int(k), int(rerank_scores.numel()))
                tk = rerank_scores.topk(k=final_k, dim=0, largest=True, sorted=False)
                ridx = tk.indices.detach().cpu().numpy().astype(np.int64)
                ids = ids[ridx]
                scores = tk.values.detach().cpu().numpy().astype(np.float32)
            ids, scores = self._maybe_reorder_by_index(ids, scores)
        else:
            ids, scores = self._search_cache(sample_uid, q_repr, topk=k, device=device)

        tok = self.cache.gather_by_indices(sample_uid, ids)
        tokens = torch.from_numpy(tok).to(device=device, dtype=torch.float32).unsqueeze(0)
        indices = torch.from_numpy(ids).to(device=device, dtype=torch.long).unsqueeze(0)
        vals = torch.from_numpy(scores).to(device=device, dtype=torch.float32).unsqueeze(0)
        return OfflineRetrievalOutput(tokens=tokens, indices=indices, scores=vals, total_tokens=total)
