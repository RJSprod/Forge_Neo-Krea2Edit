from types import SimpleNamespace

from krea2edit_ref.attention_compat import install_qwen_attention_factory_compatibility
from krea2edit_ref.state import Krea2EditJobState, cleanup_state, owned_by


def test_direct_attention_function_is_adapted_and_restored():
    def attention_pytorch(q, k, v, heads, mask=None):
        return q, k, v, heads, mask

    namespace = {"attention_function": attention_pytorch}
    exec("def forward(self): return attention_function('cpu', mask=False, small_input=True)", namespace)
    forward = namespace["forward"]
    visual = SimpleNamespace(forward=forward)
    engine = SimpleNamespace(text_processing_engine_qwen=SimpleNamespace(text_encoder=SimpleNamespace(visual=visual)))
    state = Krea2EditJobState(True, [], [], "fit", 768, 1, 1)

    install_qwen_attention_factory_compatibility(engine, state)

    installed = forward.__globals__["attention_function"]
    assert owned_by(installed, state.token)
    assert installed("cpu", mask=False, small_input=True) is attention_pytorch
    cleanup_state(state)
    assert forward.__globals__["attention_function"] is attention_pytorch


def test_attention_factory_api_is_left_unchanged():
    def attention_factory(device, mask=False, small_input=False):
        return device

    namespace = {"attention_function": attention_factory}
    exec("def forward(self): return attention_function('cpu')", namespace)
    forward = namespace["forward"]
    visual = SimpleNamespace(forward=forward)
    engine = SimpleNamespace(text_processing_engine_qwen=SimpleNamespace(text_encoder=SimpleNamespace(visual=visual)))
    state = Krea2EditJobState(True, [], [], "fit", 768, 1, 1)

    install_qwen_attention_factory_compatibility(engine, state)

    assert forward.__globals__["attention_function"] is attention_factory
