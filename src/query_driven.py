from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


class QueryDrivenCompressor(nn.Module):
    """
    Core OpticalRAG module:
    visual tokens + query latent -> query-focused visual tokens.
    """

    def __init__(
        self,
        d_vis: int = 1024,
        d_llm: int = 3584,
        num_query_tokens: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        use_depthwise_downsample: bool = True,
        downsample_stride: int = 2,
        query_pooling: str = "mean",
        pre_filter_topk: int = 0,
        pre_filter_mode: str = "hard",
        pre_filter_soft_scale: float = 1.0,
    ):
        super().__init__()
        self.d_vis = d_vis
        self.num_query_tokens = num_query_tokens
        self.use_depthwise_downsample = use_depthwise_downsample
        self.downsample_stride = downsample_stride
        self.query_pooling = query_pooling
        self.pre_filter_topk = max(0, int(pre_filter_topk))
        self.pre_filter_mode = pre_filter_mode
        self.pre_filter_soft_scale = float(pre_filter_soft_scale)

        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, d_vis) * 0.02)
        self.proj_q = nn.Linear(d_llm, d_vis, bias=False)
        self.query_attn_score = nn.Linear(d_vis, 1, bias=False)

        self.proj_k = nn.Linear(d_vis, d_vis, bias=False)
        self.proj_v = nn.Linear(d_vis, d_vis, bias=False)

        self.downsample = nn.Conv2d(
            in_channels=d_vis,
            out_channels=d_vis,
            kernel_size=downsample_stride,
            stride=downsample_stride,
            groups=d_vis,
            bias=False,
        )

        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(d_vis, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_vis)

    def project_query(self, query_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj_q(query_hidden)

    def pool_query(self, proj_q: torch.Tensor) -> torch.Tensor:
        if self.query_pooling == "none":
            return torch.zeros(
                (proj_q.shape[0], 1, proj_q.shape[-1]),
                device=proj_q.device,
                dtype=proj_q.dtype,
            )
        if self.query_pooling == "last":
            return proj_q[:, -1:, :]
        if self.query_pooling == "max":
            return proj_q.max(dim=1, keepdim=True).values
        if self.query_pooling == "max_mean":
            q_max = proj_q.max(dim=1, keepdim=True).values
            q_mean = proj_q.mean(dim=1, keepdim=True)
            return 0.5 * (q_max + q_mean)
        if self.query_pooling == "attn":
            w = torch.softmax(self.query_attn_score(proj_q), dim=1)
            return (w * proj_q).sum(dim=1, keepdim=True)
        return proj_q.mean(dim=1, keepdim=True)

    def _downsample_visual_tokens(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, N, d_vis]
        bsz, n, d = v.shape
        side = int(math.sqrt(n))
        if side * side != n or not self.use_depthwise_downsample:
            return v
        x = v.view(bsz, side, side, d).permute(0, 3, 1, 2)  # [B, d, H, W]
        x = self.downsample(x)                               # [B, d, H', W']
        x = x.flatten(2).permute(0, 2, 1)                   # [B, N', d]
        return x

    def _query_filter_visual_tokens(self, v: torch.Tensor, q_global: torch.Tensor) -> torch.Tensor:
        # v: [B, N, d_vis], q_global: [B, 1, d_vis]
        if self.pre_filter_mode == "soft":
            # Soft routing: keep all tokens and reweight by query relevance.
            scale = max(1e-6, self.pre_filter_soft_scale)
            denom = math.sqrt(max(1.0, float(v.shape[-1])))
            scores = (v * q_global).sum(dim=-1, keepdim=True) / denom
            weights = torch.sigmoid(scale * scores)
            return v * weights
        if self.pre_filter_topk <= 0:
            return v
        bsz, n, _ = v.shape
        k = min(self.pre_filter_topk, n)
        if k >= n:
            return v
        scores = (v * q_global).sum(dim=-1)  # [B, N]
        topk_idx = scores.topk(k=k, dim=1, largest=True, sorted=False).indices  # [B, k]
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, v.shape[-1])
        return torch.gather(v, dim=1, index=gather_idx)

    def forward(self, visual_tokens: torch.Tensor, query_hidden: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, N, d_vis]
        # query_hidden : [B, q, d_llm]
        bsz = visual_tokens.shape[0]
        proj_q = self.project_query(query_hidden)
        q_global = self.pool_query(proj_q)
        v = visual_tokens + q_global
        v = self._query_filter_visual_tokens(v, q_global)
        v = self._downsample_visual_tokens(v)

        latents = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1) + q_global
        k = self.proj_k(v)
        val = self.proj_v(v)

        h = latents
        for layer in self.layers:
            out, _ = layer(h, k, val, need_weights=False)
            h = h + out
        return self.norm(h)


class VisualToLLMBridge(nn.Module):
    def __init__(self, d_vis: int = 1024, d_llm: int = 3584, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_vis, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_llm),
            nn.LayerNorm(d_llm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class NoQueryDrivenCompressor(nn.Module):
    """
    Query-driven compressor ablation:
    - mean_pool16: split token sequence into m chunks and mean-pool each chunk.
    - rand_proj16: fixed random projection queries attend over visual tokens.
    """

    def __init__(
        self,
        d_vis: int = 1024,
        num_query_tokens: int = 16,
        mode: Literal["mean_pool16", "rand_proj16"] = "mean_pool16",
        rand_seed: int = 1234,
    ):
        super().__init__()
        self.d_vis = int(d_vis)
        self.num_query_tokens = int(num_query_tokens)
        self.mode = str(mode)
        self.norm = nn.LayerNorm(self.d_vis)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(rand_seed))
        rand_queries = torch.randn(self.num_query_tokens, self.d_vis, generator=g)
        rand_queries = rand_queries / rand_queries.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("rand_queries", rand_queries, persistent=True)

    def _chunk_mean_pool(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, N, D]
        bsz, n, d = visual_tokens.shape
        m = self.num_query_tokens
        idx = torch.arange(n, device=visual_tokens.device)
        # Evenly assign each token to one of m bins.
        bins = torch.clamp((idx * m) // max(1, n), max=m - 1)  # [N]
        out = []
        for j in range(m):
            mask = bins == j
            if mask.any():
                pooled = visual_tokens[:, mask, :].mean(dim=1)
            else:
                pooled = visual_tokens.mean(dim=1)
            out.append(pooled)
        return torch.stack(out, dim=1)  # [B, m, D]

    def _random_projection_pool(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, N, D]
        d = max(1.0, float(visual_tokens.shape[-1]))
        q = self.rand_queries.to(dtype=visual_tokens.dtype, device=visual_tokens.device)  # [m, D]
        # scores: [B, N, m]
        scores = torch.einsum("bnd,md->bnm", visual_tokens, q) / math.sqrt(d)
        # weights over tokens for each random query: [B, m, N]
        weights = torch.softmax(scores.transpose(1, 2), dim=-1)
        # pooled: [B, m, D]
        return torch.einsum("bmn,bnd->bmd", weights, visual_tokens)

    def forward(self, visual_tokens: torch.Tensor, query_hidden: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "rand_proj16":
            out = self._random_projection_pool(visual_tokens)
        else:
            out = self._chunk_mean_pool(visual_tokens)
        return self.norm(out)
