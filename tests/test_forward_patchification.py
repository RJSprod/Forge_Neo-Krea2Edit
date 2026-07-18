import torch
import torch.nn.functional as F
from einops import rearrange

from krea2edit_ref import forward


def _pad_to_patch_size(value, size):
    patch_h, patch_w = size
    return F.pad(value, (0, (-value.shape[-1]) % patch_w, 0, (-value.shape[-2]) % patch_h))


def test_target_and_non_square_reference_patch_shapes():
    target = torch.zeros(1, 16, 128, 128)
    reference = torch.zeros(1, 16, 128, 84)

    target_patches = rearrange(target, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    reference_patches = rearrange(reference, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    assert target_patches.shape == (1, 4096, 64)
    assert reference_patches.shape == (1, 64 * 42, 64)


class _First(torch.nn.Module):
    in_features = 64

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        assert value.ndim == 3
        assert value.shape[-1] == 64
        self.inputs.append(value.detach().clone())
        return value


class _Tmlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value)
        return value


class _Last:
    class linear:
        out_features = 64

    def __call__(self, sequence, timestep):
        return sequence


class _Model:
    patch = 2
    channels = 16
    tdim = 256

    def __init__(self):
        self.first = _First()
        self.tmlp = _Tmlp()
        self.tproj = lambda value: value
        self.txtfusion = lambda value, **kwargs: value
        self.txtmlp = lambda value: value
        self._unpack_context = lambda value: value
        self.position_ids = None
        self.pe_embedder = self._positions
        self.blocks = [lambda sequence, *args, **kwargs: sequence]
        self.last = _Last()

    def _positions(self, value):
        self.position_ids = value
        return value


def test_mock_forward_patchifies_references_embeds_timesteps_and_unpatchifies(monkeypatch):
    monkeypatch.setattr(
        forward,
        "_forge_patch_helpers",
        lambda: (lambda timesteps, width: torch.zeros(timesteps.shape[0], width), _pad_to_patch_size),
    )
    model = _Model()
    target = torch.zeros(1, 16, 127, 85)
    context = torch.zeros(1, 1, 2, 64)
    references = [torch.ones(1, 16, 128, 84), torch.full((1, 16, 64, 42), 2.0)]

    output = forward.krea2_edit_forward(
        model, target, torch.tensor([1.0]), context, references, fit_mode="fit"
    )

    assert [value.shape[-1] for value in model.first.inputs] == [64, 64, 64]
    assert model.first.inputs[0].shape == (1, 64 * 43, 64)
    assert model.first.inputs[1].shape == (1, 64 * 42, 64)
    assert model.first.inputs[2].shape == (1, 32 * 21, 64)
    assert model.tmlp.inputs[0].shape == (1, 1, model.tdim)
    assert model.position_ids.shape[1] == 2 + 64 * 42 + 32 * 21 + 64 * 43
    assert output.shape == target.shape


def test_unpatchification_round_trip_and_crop():
    tokens = torch.arange(1 * 3 * 2 * 64).reshape(1, 6, 64)
    output = rearrange(
        tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=3, w=2, c=16, ph=2, pw=2,
    )
    assert output.shape == (1, 16, 6, 4)
    assert output[..., :5, :3].shape == (1, 16, 5, 3)
