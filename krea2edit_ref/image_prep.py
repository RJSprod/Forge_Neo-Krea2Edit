from __future__ import annotations
import base64, hashlib, io
import numpy as np
import torch
from PIL import Image, ImageOps

def normalize_image(value, background=(255, 255, 255)):
    if isinstance(value, tuple): value = value[0]
    if isinstance(value, str):
        payload = value.split(",", 1)[-1] if value.startswith("data:") else value
        try: value = Image.open(io.BytesIO(base64.b64decode(payload)))
        except Exception as e: raise RuntimeError("Invalid or corrupt reference image.") from e
    if not isinstance(value, Image.Image): raise RuntimeError("Invalid or corrupt reference image.")
    try:
        image = ImageOps.exif_transpose(value)
        if image.width < 1 or image.height < 1: raise ValueError("zero size")
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            base = Image.new("RGBA", image.size, (*background, 255)); base.alpha_composite(image.convert("RGBA")); image = base.convert("RGB")
        else: image = image.convert("RGB")
        return image.copy()
    except Exception as e: raise RuntimeError("Invalid or corrupt reference image.") from e

def extract_gallery(gallery):
    refs = [normalize_image(item) for item in (gallery or [])]
    if len(refs) > 2: raise RuntimeError("A maximum of two references is supported. Use scene first, subject second.")
    return refs

def image_hash(image):
    h = hashlib.sha256(); h.update(f"{image.width}x{image.height}:RGB".encode()); h.update(image.tobytes()); return h.hexdigest()

def grounding_tensor(image, longest_side):
    if longest_side and max(image.size) > longest_side:
        ratio = longest_side / max(image.size); size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        image = image.resize(size, Image.Resampling.BOX)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)

def vae_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
