from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import DEFAULT_DEEPSEEK_OCR_PATH, resolve_deepseek_ocr_path
from .dynamic_deepseek import load_deepencoder_module, load_full_state_dict


class DeepEncoder(nn.Module):
    def __init__(self, checkpoint: str | Path | None = None):
        super().__init__()
        self.checkpoint = resolve_deepseek_ocr_path(checkpoint or DEFAULT_DEEPSEEK_OCR_PATH)
        module = load_deepencoder_module(str(self.checkpoint))
        self.sam = module.build_sam_vit_b()
        self.clip = module.build_clip_l()
        self._load_weights(self.checkpoint)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _load_weights(self, checkpoint: str | Path) -> None:
        raw = load_full_state_dict(str(checkpoint))
        sam_sd = {k[len("model.sam_model.") :]: v for k, v in raw.items() if k.startswith("model.sam_model.")}
        clip_sd = {k[len("model.vision_model.") :]: v for k, v in raw.items() if k.startswith("model.vision_model.")}
        # The official checkpoint omits this deterministic, non-learned arange buffer.
        # Restore that buffer only; strict loading still rejects missing learned weights.
        if "embeddings.position_ids" not in clip_sd:
            clip_sd["embeddings.position_ids"] = self.clip.embeddings.position_ids.detach().clone()
        self.sam.load_state_dict(sam_sd, strict=True)
        self.clip.load_state_dict(clip_sd, strict=True)

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        sam_out = self.sam(image)
        sam_flat = sam_out.flatten(2).permute(0, 2, 1)
        clip_out = self.clip(image, sam_out)
        index_token = F.normalize(clip_out[:, 0], dim=-1)
        compressed_token = sam_flat + clip_out[:, 1:]
        return index_token, compressed_token



def preprocess_image(
    img,
    size: int = 640,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    if isinstance(img, torch.Tensor):
        array = img.detach().cpu().numpy()
        if img.dim() == 3 and img.shape[0] in (1, 3):
            array = array.transpose(1, 2, 0)
        if array.dtype != np.uint8:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(array)
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)

    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    tensor = torch.from_numpy(np.asarray(img).astype("float32") / 255.0)
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)
