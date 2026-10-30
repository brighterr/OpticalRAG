from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RenderDensity:
    font_size: int
    margin: int
    line_height_scale: float

    def normalized(self) -> "RenderDensity":
        return RenderDensity(
            font_size=max(8, int(self.font_size)),
            margin=max(4, int(self.margin)),
            line_height_scale=max(1.0, float(self.line_height_scale)),
        )


@dataclass(frozen=True)
class RenderDensityRule:
    max_context_chars: int
    density: RenderDensity


DEFAULT_DENSITY = RenderDensity(font_size=16, margin=24, line_height_scale=1.45)


def _parse_density_triplet(spec: str) -> RenderDensity:
    parts = [x.strip() for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid density spec: {spec}")
    return RenderDensity(
        font_size=int(parts[0]),
        margin=int(parts[1]),
        line_height_scale=float(parts[2]),
    ).normalized()


def _parse_rules(spec: str) -> List[RenderDensityRule]:
    """
    Parse rule string:
    "300000:15,20,1.3;900000:15,22,1.35;inf:16,24,1.45"
    """
    rules: List[RenderDensityRule] = []
    for chunk in [x.strip() for x in spec.split(";") if x.strip()]:
        if ":" not in chunk:
            raise ValueError(f"Invalid rule chunk: {chunk}")
        bound_s, density_s = [x.strip() for x in chunk.split(":", 1)]
        if bound_s.lower() in {"inf", "infinity", "max"}:
            bound = 10**18
        else:
            bound = int(bound_s)
        rules.append(RenderDensityRule(max_context_chars=bound, density=_parse_density_triplet(density_s)))
    rules.sort(key=lambda x: x.max_context_chars)
    return rules


def _density_from_env(default: RenderDensity) -> RenderDensity:
    f = os.getenv("OPTICALRAG_RENDER_FONT_SIZE")
    m = os.getenv("OPTICALRAG_RENDER_MARGIN")
    lh = os.getenv("OPTICALRAG_RENDER_LINE_HEIGHT_SCALE")
    if f is None and m is None and lh is None:
        return default.normalized()
    return RenderDensity(
        font_size=int(f if f is not None else default.font_size),
        margin=int(m if m is not None else default.margin),
        line_height_scale=float(lh if lh is not None else default.line_height_scale),
    ).normalized()


class RenderDensityPolicy:
    """
    A standalone component that decides render density from context length.
    Controlled by environment variables for easy experiment iteration.

    Modes:
      - fixed (default): one density for all samples
      - length: piecewise density by context length
    """

    def __init__(self, mode: str, fixed: RenderDensity, rules: List[RenderDensityRule]):
        self.mode = mode
        self.fixed = fixed.normalized()
        self.rules = rules

    @classmethod
    def from_env(cls) -> "RenderDensityPolicy":
        mode = os.getenv("OPTICALRAG_RENDER_POLICY_MODE", "fixed").strip().lower()
        fixed = _density_from_env(DEFAULT_DENSITY)

        if mode in {"length", "adaptive", "length_adaptive"}:
            rule_spec = os.getenv("OPTICALRAG_RENDER_POLICY_RULES", "").strip()
            if rule_spec:
                rules = _parse_rules(rule_spec)
            else:
                # default length-adaptive schedule (can be overridden by OPTICALRAG_RENDER_POLICY_RULES)
                rules = [
                    RenderDensityRule(300000, RenderDensity(15, 20, 1.30).normalized()),
                    RenderDensityRule(900000, RenderDensity(15, 22, 1.35).normalized()),
                    RenderDensityRule(10**18, RenderDensity(16, 24, 1.45).normalized()),
                ]
            return cls(mode="length", fixed=fixed, rules=rules)

        return cls(mode="fixed", fixed=fixed, rules=[])

    def choose(self, context_chars: int) -> RenderDensity:
        if self.mode != "length" or not self.rules:
            return self.fixed
        n = max(0, int(context_chars))
        for rule in self.rules:
            if n <= rule.max_context_chars:
                return rule.density
        return self.rules[-1].density

