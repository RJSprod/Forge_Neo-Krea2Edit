from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Geometry:
    width: int; height: int; offset_x: int; offset_y: int; crop: bool

def reference_geometry(source_w, source_h, target_w, target_h, mode="fit", tolerance=.08):
    if mode == "crop (legacy)": return Geometry(target_w, target_h, 0, 0, True)
    source_ratio, target_ratio = source_w / source_h, target_w / target_h
    if abs(source_ratio / target_ratio - 1) <= tolerance: return Geometry(target_w, target_h, 0, 0, True)
    scale = min(target_w / source_w, target_h / source_h)
    w = max(16, min(target_w // 16 * 16, int(source_w * scale) // 16 * 16))
    h = max(16, min(target_h // 16 * 16, int(source_h * scale) // 16 * 16))
    return Geometry(w, h, (target_w // 8 - w // 8) // 2, (target_h // 8 - h // 8) // 2, False)
