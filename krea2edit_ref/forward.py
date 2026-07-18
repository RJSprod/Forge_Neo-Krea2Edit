from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .geometry import reference_geometry
from .image_prep import vae_tensor

MAX_BIAS_BYTES = 512 * 1024 * 1024

def _imgids(batch, height, width, frame=0, device=None):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack((torch.full_like(y, frame), y, x), -1).reshape(1, height * width, 3).repeat(batch, 1, 1)

def _imgids_offset(batch, height, width, frame, offset_y, offset_x, device=None):
    ids = _imgids(batch, height, width, frame, device)
    ids[..., 1] += offset_y; ids[..., 2] += offset_x
    return ids

def build_reference_bias(batch, text_len, reference_lengths, target_len, boosts, device, dtype, max_bytes=MAX_BIAS_BYTES):
    if all(abs(float(x) - 1.0) < 1e-7 for x in boosts): return None
    total = text_len + sum(reference_lengths) + target_len
    required = total * total * torch.tensor([], dtype=dtype).element_size()
    if required > max_bytes: raise RuntimeError(f"Krea2Edit reference fidelity mask would require {required / 1024**2:.1f} MiB, above the safe limit.")
    bias = torch.zeros((1, 1, total, total), device=device, dtype=dtype)
    target_start, ref_start = total - target_len, text_len
    for length, boost in zip(reference_lengths, boosts):
        if boost != 1.0: bias[:, :, target_start:, ref_start:ref_start + length] = math.log(max(float(boost), 1e-4))
        ref_start += length
    return bias

def _prepare_latents(engine, state, target):
    h, w = target.shape[-2:]; identity = state.model_identity
    output = []
    for index, (image, digest) in enumerate(zip(state.raw_references, state.reference_hashes)):
        key = (index, digest, state.fit_mode, h, w, identity)
        latent = state.latent_cache.get(key)
        if latent is None:
            g = reference_geometry(image.width, image.height, w * 8, h * 8, state.fit_mode)
            source = image
            if g.crop:
                target_ratio = g.width / g.height; ratio = source.width / source.height
                if ratio > target_ratio:
                    crop_w = round(source.height * target_ratio); left = (source.width - crop_w) // 2; source = source.crop((left, 0, left + crop_w, source.height))
                else:
                    crop_h = round(source.width / target_ratio); top = (source.height - crop_h) // 2; source = source.crop((0, top, source.width, top + crop_h))
            source = source.resize((g.width, g.height), torch_to_pil_resample())
            latent = engine.encode_first_stage(vae_tensor(source))
            state.latent_cache[key] = latent; state.tensors_allocated = True
        output.append(latent.to(device=target.device, dtype=target.dtype).expand(target.shape[0], -1, -1, -1))
    return output

def torch_to_pil_resample():
    from PIL import Image
    return Image.Resampling.BICUBIC

def krea2_edit_forward(model, x, timesteps, context, source_latents, transformer_options=None, ref_boost=1., ref_boost_a=1., **kwargs):
    """Forge Neo Krea forward with clean reference tokens inserted before target tokens."""
    transformer_options = transformer_options or {}
    temporal = x.ndim == 5
    if temporal:
        batch, channels, frames, height, width = x.shape; x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    original_h, original_w = x.shape[-2:]; patch = model.patch
    x = F.pad(x, (0, (-x.shape[-1]) % patch, 0, (-x.shape[-2]) % patch))
    target_h, target_w = x.shape[-2] // patch, x.shape[-1] // patch
    target = model.first(x).flatten(2).transpose(1, 2)
    refs = [model.first(F.pad(lat, (0, (-lat.shape[-1]) % patch, 0, (-lat.shape[-2]) % patch))).flatten(2).transpose(1, 2) for lat in source_latents]
    context = model._unpack_context(context.squeeze(1))
    t = model.tmlp(timesteps); tvec = model.tproj(t)
    context = model.txtmlp(model.txtfusion(context))
    sequence = torch.cat([context, *refs, target], dim=1)
    positions = [torch.zeros((x.shape[0], context.shape[1], 3), device=x.device, dtype=torch.long)]
    for number, tokens in enumerate(refs, 1):
        side_h = source_latents[number - 1].shape[-2] // patch; side_w = source_latents[number - 1].shape[-1] // patch
        positions.append(_imgids(x.shape[0], side_h, side_w, number, device=x.device))
    positions.append(_imgids(x.shape[0], target_h, target_w, 0, device=x.device))
    freqs = model.pe_embedder(torch.cat(positions, dim=1))
    boosts = [ref_boost] if len(refs) == 1 else [ref_boost_a, ref_boost]
    mask = build_reference_bias(x.shape[0], context.shape[1], [r.shape[1] for r in refs], target.shape[1], boosts, x.device, target.dtype)
    for block in model.blocks: sequence = block(sequence, tvec, freqs, mask, transformer_options=transformer_options)
    output = model.last(sequence, t)[:, -target.shape[1]:].transpose(1, 2)
    output = output.reshape(x.shape[0], model.channels, target_h, target_w).repeat_interleave(patch, -2).repeat_interleave(patch, -1)[..., :original_h, :original_w]
    if temporal: output = output.reshape(batch, frames, model.channels, original_h, original_w).permute(0, 2, 1, 3, 4)
    return output
