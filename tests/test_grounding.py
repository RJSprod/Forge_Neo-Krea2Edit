import pytest
import torch

from krea2edit_ref.grounding import _run_grounded, install_grounded_conditioning
from krea2edit_ref.state import Krea2EditJobState, cleanup_state


class Emphasis:
    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


class TextEngine:
    def __init__(self, z_shape, multiplier_shape):
        self.emphasis = Emphasis()
        self.z_shape = z_shape
        self.multiplier_shape = multiplier_shape
        self.images = None

    def __call__(self, prompts, images):
        self.images = images
        self.emphasis.z = torch.ones(self.z_shape)
        self.emphasis.multipliers = torch.ones(self.multiplier_shape)
        self.emphasis.after_transformers()
        return self.emphasis.z


def test_grounding_skips_only_incompatible_image_token_emphasis_and_restores_hook():
    engine = TextEngine((1, 521, 8), (1, 18))
    original = engine.emphasis.after_transformers

    output = _run_grounded(engine, ["prompt"], ["reference"])

    assert output.shape == (1, 521, 8)
    assert engine.images == ["reference"]
    assert engine.emphasis.after_transformers == original


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


def test_grounding_hook_is_restored_when_forward_installation_never_happens():
    class Engine:
        def get_learned_conditioning(self, prompts):
            return prompts

    engine = Engine()
    original = engine.get_learned_conditioning
    state = Krea2EditJobState(True, [], [], "fit", 768, 1, 1)

    install_grounded_conditioning(engine, state)
    cleanup_state(state)

    assert engine.get_learned_conditioning == original
