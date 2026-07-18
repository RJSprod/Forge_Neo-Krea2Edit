from __future__ import annotations
import math
import logging
import torch
from einops import rearrange
from .geometry import reference_geometry
from .image_prep import vae_tensor

MAX_BIAS_BYTES = 512 * 1024 * 1024
log = logging.getLogger(__name__)

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

def _to_4d(value: torch.Tensor, name: str) -> torch.Tensor:
    """Convert Forge Krea image/video latents from BCHW or BCTHW to BCHW."""
    if value.ndim == 4:
        return value
    if value.ndim == 5:
        batch, channels, frames, height, width = value.shape
        # Preserve batch/frame ordering when adapting temporal VAE output.
        return value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    raise RuntimeError(
        f"Krea2Edit expected {name} to be BCHW or BCTHW, "
        f"but received shape {tuple(value.shape)}."
    )


def _prepare_latents(engine, state, target):
    target = _to_4d(target, "target latent")
    target_batch = target.shape[0]
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
            raw_latent = engine.encode_first_stage(vae_tensor(source))
            latent = _to_4d(raw_latent, f"reference latent {index + 1}")
            log.info(
                "[Forge Krea2Edit Ref] reference %d latent: VAE shape=%s, normalized shape=%s",
                index + 1, tuple(raw_latent.shape), tuple(latent.shape),
            )
            state.latent_cache[key] = latent; state.tensors_allocated = True
        else:
            # Normalize cache entries created by an older extension version.
            latent = _to_4d(latent, f"cached reference latent {index + 1}")
            state.latent_cache[key] = latent
        latent = latent.to(device=target.device, dtype=target.dtype)
        if latent.shape[0] == 1 and target_batch > 1:
            latent = latent.expand(target_batch, *latent.shape[1:])
        elif latent.shape[0] != target_batch:
            raise RuntimeError(
                "Krea2Edit reference batch does not match the target batch: "
                f"reference={latent.shape[0]}, target={target_batch}."
            )
        output.append(latent)
    return output

def torch_to_pil_resample():
    from PIL import Image
    return Image.Resampling.BICUBIC


def _forge_patch_helpers():
    """Import Forge helpers only while executing the Forge-only forward path."""
    from backend.nn.flux import timestep_embedding
    from backend.utils import pad_to_patch_size

    return timestep_embedding, pad_to_patch_size


def _validate_patch_layout(model):
    patch = getattr(model, "patch", None)
    if not isinstance(patch, int) or isinstance(patch, bool) or patch <= 0:
        raise RuntimeError(f"Unsupported Forge Krea patch size: {patch!r}.")
    expected = model.channels * patch ** 2
    actual = getattr(model.first, "in_features", None)
    if actual is not None and actual != expected:
        raise RuntimeError(
            "Unsupported Forge Krea patch embedding layout: expected channels * "
            f"patch^2 ({expected}), but model.first reports {actual}."
        )
    linear = getattr(model.last, "linear", None)
    actual = getattr(linear, "out_features", None)
    if actual is not None and actual != expected:
        raise RuntimeError(
            "Unsupported Forge Krea output patch layout: expected channels * "
            f"patch^2 ({expected}), but model.last.linear reports {actual}."
        )
    return patch, expected


def _patchify(model, value, name, pad_to_patch_size):
    if value.ndim != 4:
        raise RuntimeError(f"Krea {name} must be BCHW; received {tuple(value.shape)}.")
    if value.shape[1] != model.channels:
        raise RuntimeError(
            f"Krea {name} channel mismatch: model expects {model.channels}, "
            f"received {value.shape[1]}."
        )
    patch, expected_width = _validate_patch_layout(model)
    padded = pad_to_patch_size(value, (patch, patch))
    height, width = padded.shape[-2] // patch, padded.shape[-1] // patch
    patches = rearrange(
        padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch
    )
    if patches.shape[-1] != expected_width:
        raise RuntimeError(
            f"Krea {name} patch width mismatch: expected {expected_width}, "
            f"received {patches.shape[-1]}."
        )
    return padded, patches, (height, width)

def krea2_edit_forward(model, x, timesteps, context, source_latents,
                       transformer_options=None, ref_boost=1., ref_boost_a=1.,
                       fit_mode="fit", **kwargs):
    """Forge Neo Krea forward with clean reference tokens inserted before target tokens."""
    transformer_options = transformer_options or {}
    temporal = x.ndim == 5
    if temporal:
        batch, channels, frames, height, width = x.shape; x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    timestep_embedding, pad_to_patch_size = _forge_patch_helpers()
    original_h, original_w = x.shape[-2:]
    patch, expected_width = _validate_patch_layout(model)
    x, target_patches, (target_h, target_w) = _patchify(
        model, x, "target", pad_to_patch_size
    )
    target = model.first(target_patches)
    refs, reference_grids, reference_patches = [], [], []
    for index, latent in enumerate(source_latents):
        padded, patches, grid = _patchify(
            model, latent, f"reference {index + 1}", pad_to_patch_size
        )
        del padded
        reference_patches.append(patches)
        refs.append(model.first(patches))
        reference_grids.append(grid)
    if not getattr(model, "_krea2edit_patch_layout_logged", False):
        log.info("[Forge Krea2Edit Ref] patch layout: channels=%d patch=%d width=%d", model.channels, patch, expected_width)
        log.info("[Forge Krea2Edit Ref] target: latent=%s patches=%s tokens=%s", tuple(x.shape), tuple(target_patches.shape), tuple(target.shape))
        for index, (patches, tokens) in enumerate(zip(reference_patches, refs), 1):
            log.info("[Forge Krea2Edit Ref] reference %d: patches=%s tokens=%s", index, tuple(patches.shape), tuple(tokens.shape))
        model._krea2edit_patch_layout_logged = True
    context = model._unpack_context(context.squeeze(1))
    timestep_input = timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(target.dtype)
    t = model.tmlp(timestep_input); tvec = model.tproj(t)
    context = model.txtmlp(model.txtfusion(context, mask=None, transformer_options=transformer_options))
    sequence = torch.cat([context, *refs, target], dim=1)
    positions = [torch.zeros((x.shape[0], context.shape[1], 3), device=x.device, dtype=torch.long)]
    for number, (side_h, side_w) in enumerate(reference_grids, 1):
        offset_y = offset_x = 0
        if fit_mode == "fit":
            offset_y = max(0, (target_h - side_h) // 2)
            offset_x = max(0, (target_w - side_w) // 2)
        positions.append(_imgids_offset(x.shape[0], side_h, side_w, number, offset_y, offset_x, x.device))
    positions.append(_imgids(x.shape[0], target_h, target_w, 0, device=x.device))
    freqs = model.pe_embedder(torch.cat(positions, dim=1))
    boosts = [ref_boost] if len(refs) == 1 else [ref_boost_a, ref_boost]
    mask = build_reference_bias(x.shape[0], context.shape[1], [r.shape[1] for r in refs], target.shape[1], boosts, x.device, target.dtype)
    for block in model.blocks: sequence = block(sequence, tvec, freqs, mask, transformer_options=transformer_options)
    target_start = context.shape[1] + sum(ref.shape[1] for ref in refs)
    target_output = model.last(sequence, t)[:, target_start:target_start + target.shape[1], :]
    output = rearrange(
        target_output, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=target_h, w=target_w, ph=patch, pw=patch, c=model.channels,
    )[..., :original_h, :original_w]
    if temporal: output = output.reshape(batch, frames, model.channels, original_h, original_w).permute(0, 2, 1, 3, 4)
    return output
