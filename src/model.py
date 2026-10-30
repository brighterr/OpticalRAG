from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_LLM_PATH, get_llm_hidden_size, resolve_runtime_device
from .encoder import DeepEncoder, preprocess_image
from .offline_retrieval import OfflineVisualRetriever
from .query_driven import NoQueryDrivenCompressor, QueryDrivenCompressor, VisualToLLMBridge
from .render import render_text_to_pages, render_text_to_pages_bytepack
from .render_density_policy import RenderDensityPolicy
from .visual_retriever import QueryAwareVisualRetriever


@dataclass
class OpticalRAGOutputs:
    loss: torch.Tensor
    logits: torch.Tensor


class OpticalRAGModel(nn.Module):
    def __init__(
        self,
        llm_path: str | None = DEFAULT_LLM_PATH,
        deepseek_path: str | None = None,
        device: str = "cuda:auto",
        llm_dtype: torch.dtype = torch.bfloat16,
        num_query_tokens: int = 16,
        context_sampling: str = "head",
        query_pooling: str = "mean",
        compressor_mode: str = "qdc",
        pre_filter_topk: int = 0,
        pre_filter_mode: str = "hard",
        pre_filter_soft_scale: float = 1.0,
        visual_window_chars: int = 24000,
        visual_num_windows: int = 1,
        visual_pages_per_window: int = 4,
        full_visual_encode: bool = False,
        full_visual_chunk_chars: int = 24000,
        full_visual_render_mode: str = "text",
        retrieval_enable: bool = False,
        retrieval_topk: int = 0,
        retrieval_topk_factor: int = 8,
        retrieval_rerank_factor: int = 4,
        retrieval_backend: str = "online",
        retrieval_score_type: str = "dot",
        retrieval_score_alpha: float = 0.25,
        retrieval_adaptive_k: bool = False,
        retrieval_adaptive_min_k: int = 64,
        retrieval_adaptive_max_k: int = 384,
        retrieval_adaptive_ref_tokens: int = 40000,
        retrieval_adaptive_alpha: float = 0.5,
        retrieval_adaptive_base_k: int = 128,
        offline_cache_dir: str = "",
        offline_index_dir: str = "",
        offline_uid_prefix: str = "",
        retrieval_preserve_order: bool = False,
        use_depthwise_downsample: bool = True,
        downsample_stride: int = 2,
    ):
        super().__init__()
        self.device_obj = resolve_runtime_device(device)
        self.llm_hidden = get_llm_hidden_size(llm_path)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            trust_remote_code=True,
            torch_dtype=llm_dtype,
        ).to(self.device_obj)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()

        self.encoder = DeepEncoder(checkpoint=deepseek_path).to(self.device_obj)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        self.qcompress = QueryDrivenCompressor(
            d_vis=1024,
            d_llm=self.llm_hidden,
            num_query_tokens=num_query_tokens,
            num_heads=8,
            num_layers=2,
            use_depthwise_downsample=use_depthwise_downsample,
            downsample_stride=downsample_stride,
            query_pooling=query_pooling,
            pre_filter_topk=pre_filter_topk,
            pre_filter_mode=pre_filter_mode,
            pre_filter_soft_scale=pre_filter_soft_scale,
        ).to(self.device_obj)
        self.compressor_mode = str(compressor_mode).strip().lower()
        if self.compressor_mode not in {"qdc", "mean_pool16", "rand_proj16"}:
            raise ValueError(f"Unsupported compressor_mode: {self.compressor_mode}")
        self.no_qdc_compressor = NoQueryDrivenCompressor(
            d_vis=1024,
            num_query_tokens=num_query_tokens,
            mode="mean_pool16" if self.compressor_mode == "mean_pool16" else "rand_proj16",
        ).to(self.device_obj)
        self.bridge = VisualToLLMBridge(d_vis=1024, d_llm=self.llm_hidden, hidden_dim=max(2048, self.llm_hidden)).to(self.device_obj)
        self.retriever = QueryAwareVisualRetriever(
            retrieval_topk=retrieval_topk,
            retrieval_topk_factor=retrieval_topk_factor,
            min_topk=num_query_tokens,
            score_type=retrieval_score_type,
            score_alpha=retrieval_score_alpha,
            adaptive_topk=retrieval_adaptive_k,
            adaptive_min_k=retrieval_adaptive_min_k,
            adaptive_max_k=retrieval_adaptive_max_k,
            adaptive_ref_tokens=retrieval_adaptive_ref_tokens,
            adaptive_alpha=retrieval_adaptive_alpha,
            adaptive_base_k=retrieval_adaptive_base_k,
        ).to(self.device_obj)

        self.llm_dtype = llm_dtype
        self.context_sampling = context_sampling
        self.query_pooling = query_pooling
        self.pre_filter_topk = max(0, int(pre_filter_topk))
        self.pre_filter_mode = pre_filter_mode
        self.pre_filter_soft_scale = float(pre_filter_soft_scale)
        self.visual_window_chars = visual_window_chars
        self.visual_num_windows = max(1, int(visual_num_windows))
        self.visual_pages_per_window = max(1, int(visual_pages_per_window))
        self.full_visual_encode = bool(full_visual_encode)
        self.full_visual_chunk_chars = max(0, int(full_visual_chunk_chars))
        self.full_visual_render_mode = str(full_visual_render_mode)
        self.retrieval_enable = bool(retrieval_enable)
        self.retrieval_rerank_factor = max(1, int(retrieval_rerank_factor))
        self.retrieval_backend = str(retrieval_backend).strip().lower()
        self.retrieval_score_type = str(retrieval_score_type).strip().lower()
        self.retrieval_score_alpha = float(max(0.0, min(1.0, retrieval_score_alpha)))
        self.retrieval_adaptive_k = bool(retrieval_adaptive_k)
        self.retrieval_adaptive_min_k = max(1, int(retrieval_adaptive_min_k))
        self.retrieval_adaptive_max_k = max(self.retrieval_adaptive_min_k, int(retrieval_adaptive_max_k))
        self.retrieval_adaptive_ref_tokens = max(1, int(retrieval_adaptive_ref_tokens))
        self.retrieval_adaptive_alpha = float(max(0.0, retrieval_adaptive_alpha))
        self.retrieval_adaptive_base_k = max(1, int(retrieval_adaptive_base_k))
        self.offline_cache_dir = str(offline_cache_dir).strip()
        self.offline_index_dir = str(offline_index_dir).strip()
        self.offline_uid_prefix = str(offline_uid_prefix).strip()
        self.retrieval_preserve_order = bool(retrieval_preserve_order)
        self.offline_retriever = None
        if self.retrieval_backend in {"offline_cache", "offline_index"} and self.offline_cache_dir:
            self.offline_retriever = OfflineVisualRetriever(
                cache_dir=self.offline_cache_dir,
                index_dir=(self.offline_index_dir or None),
                score_type=self.retrieval_score_type,
                score_alpha=self.retrieval_score_alpha,
                backend=self.retrieval_backend,
                preserve_original_order=self.retrieval_preserve_order,
            )
        self.last_retrieval_stats: dict[str, object] = {}
        self.long_adapt_chars = max(0, int(os.getenv("OPTICALRAG_LONG_ADAPT_CHARS", "0")))
        self.long_adapt_windows = max(
            self.visual_num_windows,
            int(os.getenv("OPTICALRAG_LONG_ADAPT_WINDOWS", str(self.visual_num_windows))),
        )
        self.render_density_policy = RenderDensityPolicy.from_env()

    def _pool_query_for_retrieval(self, proj_q: torch.Tensor) -> torch.Tensor:
        # "none" is a compressor-only ablation; keep retrieval query-aware.
        if self.query_pooling == "none":
            return proj_q.max(dim=1, keepdim=True).values
        return self.qcompress.pool_query(proj_q)

    @property
    def trainable_parameters(self):
        if self.compressor_mode == "qdc":
            for p in self.qcompress.parameters():
                yield p
        else:
            for p in self.qcompress.proj_q.parameters():
                yield p
            if self.query_pooling == "attn":
                for p in self.qcompress.query_attn_score.parameters():
                    yield p
        for p in self.bridge.parameters():
            yield p

    def _compress_selected_tokens(self, visual_sel: torch.Tensor, q_hidden: torch.Tensor) -> torch.Tensor:
        if self.compressor_mode == "qdc":
            compressed = self.qcompress(visual_sel, q_hidden)
        else:
            compressed = self.no_qdc_compressor(visual_sel, q_hidden)
        self.last_retrieval_stats["prefix_tokens"] = int(compressed.shape[1])
        return compressed

    def _sample_context(self, context: str, max_chars: int) -> str:
        text = (context or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        if self.context_sampling == "head_tail":
            head = max_chars // 2
            tail = max_chars - head
            return text[:head] + "\n\n[...snip...]\n\n" + text[-tail:]
        return text[:max_chars]

    def _sample_context_windows(self, context: str, window_chars: int, num_windows: int) -> list[str]:
        text = (context or "").strip()
        if not text:
            return [""]
        if window_chars <= 0 or len(text) <= window_chars:
            return [text]

        max_start = max(0, len(text) - window_chars)
        starts: list[int]
        if self.context_sampling == "uniform":
            if num_windows <= 1:
                starts = [0]
            else:
                starts = [int(round(i * max_start / (num_windows - 1))) for i in range(num_windows)]
        elif self.context_sampling == "head_tail":
            if num_windows <= 1:
                starts = [0]
            else:
                starts = [0, max_start]
                remain = num_windows - 2
                if remain > 0 and max_start > 0:
                    mids = [int(round((i + 1) * max_start / (remain + 1))) for i in range(remain)]
                    starts.extend(mids)
        else:
            starts = [0]

        uniq = []
        seen = set()
        for s in starts:
            s = max(0, min(max_start, int(s)))
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return [text[s : s + window_chars] for s in uniq]

    def _split_context_full(self, context: str, chunk_chars: int) -> list[str]:
        text = (context or "").strip()
        if not text:
            return [""]
        if chunk_chars <= 0:
            return [text]
        return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    def _choose_coverage(self, context: str) -> tuple[int, int]:
        num_windows = self.visual_num_windows
        pages_per_window = self.visual_pages_per_window

        # Optional long-context adaptive coverage (disabled by default).
        if self.long_adapt_chars > 0 and self.context_sampling == "uniform":
            ctx_len = len((context or "").strip())
            if ctx_len >= self.long_adapt_chars:
                num_windows = self.long_adapt_windows
        return num_windows, pages_per_window

    def _build_visual_tokens(self, context: str) -> torch.Tensor:
        context_chars = len((context or "").strip())
        density = self.render_density_policy.choose(context_chars)
        page_batch = 4
        comps = []

        if self.full_visual_encode:
            chunk_chars = self.full_visual_chunk_chars if self.full_visual_chunk_chars > 0 else self.visual_window_chars
            chunks = self._split_context_full(context=context, chunk_chars=chunk_chars)
            for chunk in chunks:
                if self.full_visual_render_mode == "bytepack":
                    pages = render_text_to_pages_bytepack(
                        chunk,
                        page_size=640,
                        max_pages=0,
                        max_chars=0,
                    )
                else:
                    pages = render_text_to_pages(
                        chunk,
                        font_size=density.font_size,
                        margin=density.margin,
                        line_height_scale=density.line_height_scale,
                        max_pages=0,
                        max_chars=0,
                    )
                for i in range(0, len(pages), page_batch):
                    ims = [
                        preprocess_image(p, size=640, device=self.device_obj, dtype=torch.float32)
                        for p in pages[i : i + page_batch]
                    ]
                    x = torch.cat(ims, dim=0)
                    with torch.no_grad():
                        _, comp = self.encoder(x)
                    comps.append(comp.float())
        else:
            num_windows, pages_per_window = self._choose_coverage(context)
            windows = self._sample_context_windows(
                context=context,
                window_chars=self.visual_window_chars,
                num_windows=num_windows,
            )
            pages = []
            for w in windows:
                pages.extend(
                    render_text_to_pages(
                        w,
                        font_size=density.font_size,
                        margin=density.margin,
                        line_height_scale=density.line_height_scale,
                        max_pages=pages_per_window,
                        max_chars=self.visual_window_chars,
                    )
                )
            for i in range(0, len(pages), page_batch):
                ims = [
                    preprocess_image(p, size=640, device=self.device_obj, dtype=torch.float32)
                    for p in pages[i : i + page_batch]
                ]
                x = torch.cat(ims, dim=0)
                with torch.no_grad():
                    _, comp = self.encoder(x)
                comps.append(comp.float())

        comp = torch.cat(comps, dim=0)  # [P,100,1024]
        return comp.reshape(1, -1, comp.shape[-1])

    def _resolve_sample_uid(self, sample_uid: str | int | None) -> str | None:
        if sample_uid is None:
            return None
        if isinstance(sample_uid, int):
            if self.offline_uid_prefix:
                return f"{self.offline_uid_prefix}_{sample_uid:06d}"
            return str(sample_uid)
        uid = str(sample_uid).strip()
        if not uid:
            return None
        return uid

    def _retrieve_visual_tokens_online(self, visual: torch.Tensor, q_hidden: torch.Tensor) -> torch.Tensor:
        if not self.retrieval_enable:
            self.last_retrieval_stats = {
                "enabled": 0,
                "total_tokens": int(visual.shape[1]),
                "selected_tokens": int(visual.shape[1]),
            }
            return visual
        proj_q = self.qcompress.project_query(q_hidden)
        q_global = self._pool_query_for_retrieval(proj_q)
        multi_query_scores = {"maxsim", "maxsim_cosine", "dot_multi", "cosine_multi", "l2_multi"}
        if self.retrieval_score_type == "dot_hybrid_max":
            query_repr = torch.cat([q_global, proj_q], dim=1)
        elif self.retrieval_score_type in multi_query_scores:
            query_repr = proj_q
        else:
            query_repr = q_global
        out = self.retriever(visual_tokens=visual, query_repr=query_repr, num_query_tokens=self.qcompress.num_query_tokens)
        self.last_retrieval_stats = {
            "enabled": 1,
            "total_tokens": int(out.total_tokens),
            "selected_tokens": int(out.topk),
        }
        return out.tokens

    def _retrieve_visual_tokens_offline(self, q_hidden: torch.Tensor, sample_uid: str) -> torch.Tensor:
        if self.offline_retriever is None:
            raise RuntimeError("Offline retrieval backend requested but offline cache/index is not configured")
        if not self.retrieval_enable:
            raise RuntimeError("Offline retrieval backend requires retrieval_enable=1")

        proj_q = self.qcompress.project_query(q_hidden)
        q_global = self._pool_query_for_retrieval(proj_q)
        total_tokens = int(self.offline_retriever.cache.total_tokens(sample_uid))
        default_k = max(self.qcompress.num_query_tokens, self.qcompress.num_query_tokens * self.retriever.retrieval_topk_factor)
        if self.retrieval_adaptive_k:
            base_k = max(self.retrieval_adaptive_base_k, self.qcompress.num_query_tokens * self.retriever.retrieval_topk_factor)
            ratio = float(total_tokens) / float(self.retrieval_adaptive_ref_tokens)
            raw_k = float(base_k) * (max(1e-6, ratio) ** self.retrieval_adaptive_alpha)
            adaptive_k = int(round(raw_k))
            k = max(self.retrieval_adaptive_min_k, min(self.retrieval_adaptive_max_k, adaptive_k))
            k = max(self.qcompress.num_query_tokens, min(total_tokens, k))
        else:
            k = int(self.retriever.retrieval_topk) if int(self.retriever.retrieval_topk) > 0 else int(default_k)
        multi_query_scores = {"maxsim", "maxsim_cosine", "dot_multi", "cosine_multi", "l2_multi"}
        if self.retrieval_score_type == "dot_hybrid_max":
            query_repr = torch.cat([q_global, proj_q], dim=1)
        elif self.retrieval_score_type in multi_query_scores:
            query_repr = proj_q
        else:
            query_repr = q_global
        out = self.offline_retriever.retrieve(
            sample_uid=sample_uid,
            query_global=q_global,
            query_repr=query_repr,
            topk=k,
            device=self.device_obj,
            recall_topk_factor=self.retrieval_rerank_factor,
        )
        self.last_retrieval_stats = {
            "enabled": 1,
            "total_tokens": int(out.total_tokens),
            "selected_tokens": int(out.tokens.shape[1]),
            "resolved_topk": int(k),
            "adaptive_k": int(self.retrieval_adaptive_k),
            "actual_backend": self.offline_retriever.last_backend,
            "sample_uid": str(sample_uid),
            "selected_indices": out.indices.detach().cpu().reshape(-1).tolist(),
            "selected_scores": out.scores.detach().float().cpu().reshape(-1).tolist(),
            "query_sha256": hashlib.sha256(query_repr.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "all_candidates_selected": bool(out.tokens.shape[1] == out.total_tokens),
        }
        return out.tokens

    def _select_visual_tokens(self, context: str, q_hidden: torch.Tensor, sample_uid: str | int | None = None) -> torch.Tensor:
        uid = self._resolve_sample_uid(sample_uid)
        if self.retrieval_backend in {"offline_cache", "offline_index"} and uid is not None:
            return self._retrieve_visual_tokens_offline(q_hidden=q_hidden, sample_uid=uid)

        visual = self._build_visual_tokens(context)
        return self._retrieve_visual_tokens_online(visual=visual, q_hidden=q_hidden)

    def _query_hidden(self, question_prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(question_prompt, return_tensors="pt", truncation=True, max_length=1024)
        tok = {k: v.to(self.device_obj) for k, v in tok.items()}
        with torch.no_grad():
            out = self.llm(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        return out.hidden_states[-1].float(), tok["input_ids"], tok["attention_mask"]

    def _choice_token_id(self, choice: str) -> int:
        candidates = [f" {choice}", choice]
        for c in candidates:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        ids = self.tokenizer.encode(choice, add_special_tokens=False)
        return ids[0]

    def forward_sample(self, context: str, question: str, answer: str, sample_uid: str | int | None = None) -> OpticalRAGOutputs:
        # Build question prompt
        question_prompt = f"Question: {question}\nAnswer:"
        q_hidden, q_ids, q_attn = self._query_hidden(question_prompt)

        # Query-driven compression + bridge
        visual_sel = self._select_visual_tokens(context=context, q_hidden=q_hidden, sample_uid=sample_uid)
        vq = self._compress_selected_tokens(visual_sel, q_hidden)  # [1,m,1024]
        prefix = self.bridge(vq).to(self.llm_dtype)  # [1,m,d_llm]

        # Build answer ids
        ans_tok = self.tokenizer(answer, return_tensors="pt", truncation=True, max_length=128)
        ans_ids = ans_tok["input_ids"].to(self.device_obj)
        ans_attn = ans_tok["attention_mask"].to(self.device_obj)

        # Compose full embeddings: [prefix | question | answer]
        embed = self.llm.get_input_embeddings()
        q_emb = embed(q_ids).to(self.llm_dtype)
        a_emb = embed(ans_ids).to(self.llm_dtype)
        inputs_embeds = torch.cat([prefix, q_emb, a_emb], dim=1)

        prefix_mask = torch.ones((1, prefix.shape[1]), device=self.device_obj, dtype=torch.long)
        attn = torch.cat([prefix_mask, q_attn.long(), ans_attn.long()], dim=1)

        labels = torch.full((1, inputs_embeds.shape[1]), -100, device=self.device_obj, dtype=torch.long)
        start = prefix.shape[1] + q_ids.shape[1]
        labels[:, start : start + ans_ids.shape[1]] = ans_ids

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels, return_dict=True)
        return OpticalRAGOutputs(loss=out.loss, logits=out.logits)

    @torch.no_grad()
    def predict_mcq(
        self,
        context: str,
        question: str,
        choices: list[str],
        max_new_tokens: int = 24,
        sample_uid: str | int | None = None,
    ) -> str:
        q_text = (
            f"Please answer this multiple-choice question.\n"
            f"Question: {question}\n"
            f"(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n"
            f"Respond with: The correct answer is (A/B/C/D)."
        )
        q_hidden, q_ids, q_attn = self._query_hidden(q_text)
        visual_sel = self._select_visual_tokens(context=context, q_hidden=q_hidden, sample_uid=sample_uid)
        vq = self._compress_selected_tokens(visual_sel, q_hidden)
        prefix = self.bridge(vq).to(self.llm_dtype)

        embed = self.llm.get_input_embeddings()
        q_emb = embed(q_ids).to(self.llm_dtype)
        inputs_embeds = torch.cat([prefix, q_emb], dim=1)

        prefix_mask = torch.ones((1, prefix.shape[1]), device=self.device_obj, dtype=torch.long)
        attn = torch.cat([prefix_mask, q_attn.long()], dim=1)

        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text

    @torch.no_grad()
    def predict_mcq_choice(
        self,
        context: str,
        question: str,
        choices: list[str],
        sample_uid: str | int | None = None,
    ) -> str:
        return self.predict_choice_label(
            context=context,
            question=question,
            option_labels=["A", "B", "C", "D"],
            options=choices,
            sample_uid=sample_uid,
        )

    @torch.no_grad()
    def predict_choice_label(
        self,
        context: str,
        question: str,
        option_labels: list[str],
        options: list[str],
        sample_uid: str | int | None = None,
    ) -> str:
        if len(option_labels) != len(options):
            raise ValueError("option_labels and options must have the same length")
        option_block = "\n".join(f"({lab}) {opt}" for lab, opt in zip(option_labels, options))
        q_text = (
            "Please answer this multiple-choice question.\n"
            f"Question: {question}\n"
            f"{option_block}\n"
            "Output only one option label in parentheses.\n"
            "The correct answer is ("
        )
        q_hidden, q_ids, q_attn = self._query_hidden(q_text)
        visual_sel = self._select_visual_tokens(context=context, q_hidden=q_hidden, sample_uid=sample_uid)
        vq = self._compress_selected_tokens(visual_sel, q_hidden)
        prefix = self.bridge(vq).to(self.llm_dtype)

        embed = self.llm.get_input_embeddings()
        q_emb = embed(q_ids).to(self.llm_dtype)
        inputs_embeds = torch.cat([prefix, q_emb], dim=1)

        prefix_mask = torch.ones((1, prefix.shape[1]), device=self.device_obj, dtype=torch.long)
        attn = torch.cat([prefix_mask, q_attn.long()], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, return_dict=True)
        logits = out.logits[0, -1]
        ids = {lab: self._choice_token_id(lab) for lab in option_labels}
        best = max(ids.items(), key=lambda kv: float(logits[kv[1]].item()))[0]
        return best

    @torch.no_grad()
    def predict_choice_letter(
        self,
        context: str,
        question: str,
        sample_uid: str | int | None = None,
    ) -> str:
        q_text = (
            "Please answer this multiple-choice question.\n"
            f"Question: {question}\n"
            "Choose one option among A, B, C, and D.\n"
            "Output only one option label in parentheses.\n"
            "The correct answer is ("
        )
        q_hidden, q_ids, q_attn = self._query_hidden(q_text)
        visual_sel = self._select_visual_tokens(context=context, q_hidden=q_hidden, sample_uid=sample_uid)
        vq = self._compress_selected_tokens(visual_sel, q_hidden)
        prefix = self.bridge(vq).to(self.llm_dtype)

        embed = self.llm.get_input_embeddings()
        q_emb = embed(q_ids).to(self.llm_dtype)
        inputs_embeds = torch.cat([prefix, q_emb], dim=1)

        prefix_mask = torch.ones((1, prefix.shape[1]), device=self.device_obj, dtype=torch.long)
        attn = torch.cat([prefix_mask, q_attn.long()], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, return_dict=True)
        logits = out.logits[0, -1]
        option_labels = ["A", "B", "C", "D"]
        ids = {lab: self._choice_token_id(lab) for lab in option_labels}
        best = max(ids.items(), key=lambda kv: float(logits[kv[1]].item()))[0]
        return best
