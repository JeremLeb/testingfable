"""Modality specifications so encoders/decoders can dispatch by type.

A modality is either a flat vector or an egocentric image. `spaces` in the
sensor suite may hand us a bare int (legacy vector dim) or a ModalitySpec;
`normalize_spaces` coerces everything to specs so the rest of the model only
deals with one representation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalitySpec:
    shape: tuple            # vector: (dim,)  image: (C, H, W)
    kind: str = "vector"    # "vector" | "image"

    @property
    def dim(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


def normalize_spaces(spaces: dict) -> dict[str, ModalitySpec]:
    out = {}
    for name, spec in spaces.items():
        if isinstance(spec, ModalitySpec):
            out[name] = spec
        elif isinstance(spec, int):
            out[name] = ModalitySpec((spec,), "vector")
        else:  # a raw shape tuple -> infer kind by rank
            shape = tuple(spec)
            kind = "image" if len(shape) == 3 else "vector"
            out[name] = ModalitySpec(shape, kind)
    return out
