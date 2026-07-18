from types import SimpleNamespace
import re

from PIL import Image
import pytest
import torch

from krea2edit_ref.forward import _prepare_latents, _to_4d


def test_to_4d_flattens_qwen_still_image_temporal_dimension():
    latent = torch.zeros(1, 16, 1, 128, 84)

    normalized = _to_4d(latent, "reference latent 1")

    assert normalized.shape == (1, 16, 128, 84)


def test_to_4d_leaves_bchw_tensor_unchanged():
    latent = torch.zeros(1, 16, 128, 84)

    assert _to_4d(latent, "reference latent 1") is latent


@pytest.mark.parametrize("shape", [(1, 16, 84), (1, 16, 1, 2, 128, 84)])
def test_to_4d_reports_invalid_rank_and_shape(shape):
    with pytest.raises(RuntimeError, match=re.escape(f"shape {shape}")):
        _to_4d(torch.zeros(shape), "reference latent 1")


def test_prepare_latents_normalizes_caches_and_expands_references(monkeypatch):
    monkeypatch.setattr("krea2edit_ref.forward.vae_tensor", lambda image: image)

    class Engine:
        calls = 0

        def encode_first_stage(self, image):
            self.calls += 1
            return torch.zeros(1, 16, 1, 8, 8)

    state = SimpleNamespace(
        raw_references=[Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))],
        reference_hashes=["scene", "subject"],
        fit_mode="fit", model_identity=("model",), latent_cache={}, tensors_allocated=False,
    )
    target = torch.zeros(2, 16, 8, 8)
    engine = Engine()

    first = _prepare_latents(engine, state, target)
    second = _prepare_latents(engine, state, target)

    assert [latent.shape for latent in first] == [(2, 16, 8, 8), (2, 16, 8, 8)]
    assert [latent.shape for latent in second] == [(2, 16, 8, 8), (2, 16, 8, 8)]
    assert engine.calls == 2
    assert all(latent.ndim == 4 for latent in state.latent_cache.values())
