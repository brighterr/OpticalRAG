from __future__ import annotations

import math

import torch
from torch import nn


class QueryVisualScorer(nn.Module):
    """
    Standalone module for query-visual similarity.
    This is the only place that defines how (query, visual_token) are scored.
    """

    def __init__(self, score_type: str = "dot", score_alpha: float = 0.25):
        super().__init__()
        s = str(score_type).strip().lower()
        self.score_type = s
        self.score_alpha = float(max(0.0, min(1.0, score_alpha)))
        self.is_maxsim = s in {"maxsim", "maxsim_cosine"}
        self.is_multi_mean = s in {"dot_multi", "cosine_multi", "l2_multi"}
        self.is_hybrid_max = s in {"dot_hybrid_max"}
        if s == "maxsim":
            self.base_score_type = "dot"
        elif s == "maxsim_cosine":
            self.base_score_type = "cosine"
        elif s == "dot_multi":
            self.base_score_type = "dot"
        elif s == "cosine_multi":
            self.base_score_type = "cosine"
        elif s == "l2_multi":
            self.base_score_type = "l2"
        elif s == "dot_hybrid_max":
            self.base_score_type = "dot"
        else:
            self.base_score_type = s

    def _normalize(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)

    def prepare_query(self, query_repr: torch.Tensor) -> torch.Tensor:
        # query_repr: [B, Q, D]
        if self.base_score_type == "cosine":
            return self._normalize(query_repr)
        return query_repr

    def prepare_visual(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, N, D]
        if self.base_score_type == "cosine":
            return self._normalize(visual_tokens)
        return visual_tokens

    def _pair_score(self, visual_tokens: torch.Tensor, query_repr: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, N, D], query_repr: [B, Q, D]
        if self.base_score_type in {"dot", "cosine"}:
            denom = math.sqrt(float(max(1, visual_tokens.shape[-1])))
            return torch.matmul(visual_tokens, query_repr.transpose(1, 2)) / denom
        if self.base_score_type == "l2":
            v = visual_tokens.unsqueeze(2)  # [B, N, 1, D]
            q = query_repr.unsqueeze(1)     # [B, 1, Q, D]
            return -((v - q) ** 2).sum(dim=-1)
        raise ValueError(f"Unsupported score_type: {self.score_type}")

    def score(self, visual_tokens: torch.Tensor, query_repr: torch.Tensor) -> torch.Tensor:
        """
        Returns score matrix [B, N].
        """
        v = self.prepare_visual(visual_tokens)
        q = self.prepare_query(query_repr)
        pair = self._pair_score(v, q)  # [B, N, Q]

        if self.is_hybrid_max:
            if q.shape[1] <= 1:
                return pair.squeeze(-1)
            s_global = pair[:, :, 0]
            s_multi = pair[:, :, 1:].max(dim=-1).values
            a = self.score_alpha
            return (1.0 - a) * s_global + a * s_multi

        # Standard single-query scoring path.
        if q.shape[1] == 1 and not (self.is_maxsim or self.is_multi_mean):
            return pair.squeeze(-1)

        # MaxSim path for multi-query retrieval.
        if self.is_maxsim:
            return pair.max(dim=-1).values

        # Explicit multi-vector mean score.
        if self.is_multi_mean:
            return pair.mean(dim=-1)

        # Fallback for multi-query + non-maxsim score types.
        return pair.mean(dim=-1)
