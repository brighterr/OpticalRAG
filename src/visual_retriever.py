from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .query_visual_scoring import QueryVisualScorer


@dataclass
class VisualRetrieverOutput:
    tokens: torch.Tensor
    indices: torch.Tensor
    scores: torch.Tensor
    topk: int
    total_tokens: int


class QueryAwareVisualRetriever(nn.Module):
    """
    Query-aware visual-token retriever.
    Given a large visual-token pool, retrieve TopK tokens for downstream compressor.
    """

    def __init__(
        self,
        retrieval_topk: int = 0,
        retrieval_topk_factor: int = 8,
        min_topk: int = 16,
        score_type: str = "dot",
        score_alpha: float = 0.25,
        adaptive_topk: bool = False,
        adaptive_min_k: int = 64,
        adaptive_max_k: int = 384,
        adaptive_ref_tokens: int = 40000,
        adaptive_alpha: float = 0.5,
        adaptive_base_k: int = 128,
    ):
        super().__init__()
        self.retrieval_topk = int(retrieval_topk)
        self.retrieval_topk_factor = max(1, int(retrieval_topk_factor))
        self.min_topk = max(1, int(min_topk))
        self.scorer = QueryVisualScorer(score_type=score_type, score_alpha=score_alpha)
        self.adaptive_topk = bool(adaptive_topk)
        self.adaptive_min_k = max(1, int(adaptive_min_k))
        self.adaptive_max_k = max(self.adaptive_min_k, int(adaptive_max_k))
        self.adaptive_ref_tokens = max(1, int(adaptive_ref_tokens))
        self.adaptive_alpha = float(max(0.0, adaptive_alpha))
        self.adaptive_base_k = max(1, int(adaptive_base_k))

    def _resolve_topk(self, total_tokens: int, num_query_tokens: int) -> int:
        if total_tokens <= 0:
            return 1
        if self.adaptive_topk:
            base_k = max(self.adaptive_base_k, int(num_query_tokens) * self.retrieval_topk_factor)
            ratio = float(total_tokens) / float(self.adaptive_ref_tokens)
            raw_k = float(base_k) * (max(1e-6, ratio) ** self.adaptive_alpha)
            k = int(round(raw_k))
            k = max(self.adaptive_min_k, min(self.adaptive_max_k, k))
            return min(total_tokens, max(self.min_topk, k))
        if self.retrieval_topk > 0:
            return min(total_tokens, self.retrieval_topk)
        target = max(self.min_topk, int(num_query_tokens) * self.retrieval_topk_factor)
        return min(total_tokens, target)

    def forward(self, visual_tokens: torch.Tensor, query_repr: torch.Tensor, num_query_tokens: int) -> VisualRetrieverOutput:
        # visual_tokens: [B, N, d_vis]
        # query_repr: [B, Q, d_vis]
        bsz, n, d = visual_tokens.shape
        k = self._resolve_topk(total_tokens=n, num_query_tokens=num_query_tokens)
        if k >= n:
            idx = torch.arange(n, device=visual_tokens.device).view(1, n).expand(bsz, -1)
            scores = torch.ones((bsz, n), dtype=visual_tokens.dtype, device=visual_tokens.device)
            return VisualRetrieverOutput(
                tokens=visual_tokens,
                indices=idx,
                scores=scores,
                topk=n,
                total_tokens=n,
            )

        score = self.scorer.score(visual_tokens, query_repr)  # [B, N]
        topk = score.topk(k=k, dim=1, largest=True, sorted=False)
        gather_idx = topk.indices.unsqueeze(-1).expand(-1, -1, d)
        selected = torch.gather(visual_tokens, dim=1, index=gather_idx)
        return VisualRetrieverOutput(
            tokens=selected,
            indices=topk.indices,
            scores=topk.values,
            topk=k,
            total_tokens=n,
        )
