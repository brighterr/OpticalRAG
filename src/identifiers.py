"""Stable, filesystem-safe sample identifiers shared by cache and evaluation."""


def cache_sample_uid(value: object) -> str:
    uid = str(value).strip().replace("/", "__").replace("\\", "__")
    if uid in {"", ".", ".."}:
        raise ValueError(f"Invalid sample identifier: {uid!r}")
    return uid
