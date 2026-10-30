from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    configured = os.getenv("OPTICALRAG_RENDER_FONT_PATH")
    candidates = [configured] if configured else [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    raise FileNotFoundError("DejaVuSansMono.ttf missing; set OPTICALRAG_RENDER_FONT_PATH")



def render_text_to_pages(
    text: str,
    page_size: int = 640,
    font_size: int = 16,
    margin: int = 24,
    line_height_scale: float = 1.45,
    max_pages: int = 4,
    max_chars: int = 24000,
) -> List[Image.Image]:
    font_size = max(8, font_size)
    margin = max(4, margin)
    line_height_scale = max(1.0, line_height_scale)

    text = (text or "").strip()
    if not text:
        text = "(empty context)"
    if max_chars > 0:
        text = text[:max_chars]

    font = _load_font(font_size)
    line_height = int(font_size * line_height_scale)
    usable_w = page_size - 2 * margin
    usable_h = page_size - 2 * margin
    lines_per_page = max(1, usable_h // line_height)

    # estimate chars per line with monospaced-ish font
    avg_char_w = max(6, int(font_size * 0.6))
    chars_per_line = max(20, usable_w // avg_char_w)

    words = text.replace("\t", " ").split(" ")
    lines = []
    cur = ""
    for w in words:
        if not w:
            continue
        candidate = w if not cur else f"{cur} {w}"
        if len(candidate) <= chars_per_line:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            # hard wrap very long token
            while len(w) > chars_per_line:
                lines.append(w[:chars_per_line])
                w = w[chars_per_line:]
            cur = w
    if cur:
        lines.append(cur)

    pages = []
    max_pages_eff = max_pages if max_pages and max_pages > 0 else 10**9
    for pidx in range(max_pages_eff):
        start = pidx * lines_per_page
        if start >= len(lines):
            break
        end = min(len(lines), start + lines_per_page)
        img = Image.new("RGB", (page_size, page_size), color="white")
        draw = ImageDraw.Draw(img)
        y = margin
        for line in lines[start:end]:
            draw.text((margin, y), line, fill="black", font=font)
            y += line_height
        pages.append(img)

    if not pages:
        pages = [Image.new("RGB", (page_size, page_size), color="white")]
    return pages


def render_text_to_pages_bytepack(
    text: str,
    page_size: int = 640,
    max_pages: int = 0,
    max_chars: int = 0,
) -> List[Image.Image]:
    """
    High-density text-to-image packing:
    - encode text bytes
    - pack bytes into page_size*page_size grayscale pixels
    - convert to RGB pages

    This keeps full coverage while greatly reducing page count on very long contexts.
    """
    text = (text or "").strip()
    if not text:
        text = "(empty context)"
    if max_chars > 0:
        text = text[:max_chars]

    raw = text.encode("utf-8", errors="ignore")
    capacity = page_size * page_size
    if capacity <= 0:
        capacity = 640 * 640
    pages = []
    max_pages_eff = max_pages if max_pages and max_pages > 0 else 10**9

    for pidx in range(max_pages_eff):
        s = pidx * capacity
        if s >= len(raw):
            break
        e = min(len(raw), s + capacity)
        chunk = raw[s:e]
        if len(chunk) < capacity:
            chunk = chunk + bytes(capacity - len(chunk))
        img = Image.frombytes("L", (page_size, page_size), chunk).convert("RGB")
        pages.append(img)

    if not pages:
        pages = [Image.new("RGB", (page_size, page_size), color="white")]
    return pages
