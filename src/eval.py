from __future__ import annotations

import re


def extract_choice(text: str) -> str:
    if not text:
        return ""
    for p in [
        r"The correct answer is\s*\(([A-D])\)",
        r"The correct answer is\s*([A-D])",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ]:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def gold_choice(answer: str) -> str:
    m = re.findall(r"[A-D]", (answer or "").upper())
    return m[-1] if m else ""
