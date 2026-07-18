import pytest
import torch

from krea2edit_ref.grounding import _run_grounded, install_grounded_conditioning
from krea2edit_ref.state import Krea2EditJobState, cleanup_state


class Emphasis:
    def after_transformers(self):
        self.after_transformers_called = True
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


class TextEngine:
    def __init__(self, z_shape, multiplier_shape, multiplier_value=1.0):
        self.emphasis = Emphasis()
        self.z_shape = z_shape
        self.multiplier_shape = multiplier_shape
        self.multiplier_value = multiplier_value
        self.images = None

    def __call__(self, prompts, images):
        self.images = images
        # Forge's Qwen engine creates the emphasis object here, rather than
        # retaining the instance visible before conditioning starts.
        self.emphasis = Emphasis()
        self.emphasis.z = torch.ones(self.z_shape)
        self.emphasis.multipliers = torch.full(self.multiplier_shape, self.multiplier_value)
        self.emphasis.after_transformers()
        return self.emphasis.z


def test_grounding_skips_only_incompatible_image_token_emphasis_and_restores_hook():
    engine = TextEngine((1, 521, 8), (1, 18))
    original = Emphasis.after_transformers

    output = _run_grounded(engine, ["prompt"], ["reference"])

    assert output.shape == (1, 521, 8)
    assert engine.images == ["reference"]
    assert engine.emphasis.__class__ is Emphasis
    assert engine.emphasis.after_transformers_called
    assert Emphasis.after_transformers is original


def test_grounding_retains_normal_prompt_emphasis():
    engine = TextEngine((1, 18, 8), (1, 18), multiplier_value=1.2)

    output = _run_grounded(engine, ["prompt"], ["reference"])

    assert output.shape == (1, 18, 8)
    assert torch.all(output == 1.2)


def test_grounding_does_not_hide_unrelated_emphasis_errors():
    class BrokenEmphasis:
        z = torch.ones((1, 18, 8))
        multipliers = torch.ones((1, 18))

        def after_transformers(self):
            raise RuntimeError("unexpected failure")

    class BrokenEngine:
        def __init__(self):
            self.emphasis = BrokenEmphasis()

        def __call__(self, prompts, images):
            self.emphasis = BrokenEmphasis()
            self.emphasis.after_transformers()

    original = BrokenEmphasis.after_transformers
    engine = BrokenEngine()
    with pytest.raises(RuntimeError, match="unexpected failure"):
        _run_grounded(engine, ["prompt"], ["reference"])

    assert BrokenEmphasis.after_transformers is original


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
