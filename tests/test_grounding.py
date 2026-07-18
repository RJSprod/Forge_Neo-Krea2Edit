import pytest
import torch

from krea2edit_ref.grounding import _run_grounded


class Emphasis:
    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


class TextEngine:
    def __init__(self, z_shape, multiplier_shape, replace_emphasis=False):
        self.emphasis = Emphasis()
        self.z_shape = z_shape
        self.multiplier_shape = multiplier_shape
        self.replace_emphasis = replace_emphasis
        self.images = None

    def __call__(self, prompts, images):
        self.images = images
        if self.replace_emphasis:
            self.emphasis = Emphasis()
        self.emphasis.z = torch.ones(self.z_shape)
        self.emphasis.multipliers = torch.ones(self.multiplier_shape)
        self.emphasis.after_transformers()
        return self.emphasis.z


def test_grounding_skips_only_incompatible_image_token_emphasis_and_restores_hook():
    engine = TextEngine((1, 521, 8), (1, 18), replace_emphasis=True)
    original = Emphasis.after_transformers

    output = _run_grounded(engine, ["prompt"], ["reference"])

    assert output.shape == (1, 521, 8)
    assert engine.images == ["reference"]
    assert Emphasis.after_transformers is original


def test_grounding_retains_normal_prompt_emphasis():
    engine = TextEngine((1, 18, 8), (1, 18))

    output = _run_grounded(engine, ["prompt"], ["reference"])

    assert output.shape == (1, 18, 8)


def test_grounding_does_not_hide_unrelated_emphasis_errors():
    class BrokenEmphasis:
        z = torch.ones((1, 18, 8))
        multipliers = torch.ones((1, 18))

        def after_transformers(self):
            raise RuntimeError("unexpected failure")

    class BrokenEngine:
        emphasis = BrokenEmphasis()

        def __call__(self, prompts, images):
            self.emphasis.after_transformers()

    with pytest.raises(RuntimeError, match="unexpected failure"):
        _run_grounded(BrokenEngine(), ["prompt"], ["reference"])
