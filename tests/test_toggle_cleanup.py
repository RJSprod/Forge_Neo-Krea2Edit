from krea2edit_ref.state import Krea2EditJobState, cleanup_state
class Model:
 def forward(self): return 1
class Engine:
 def get_learned_conditioning(self): return 1
def test_cleanup_is_idempotent():
 state=Krea2EditJobState(True, [], [], "fit", 768, 1, 1)
 cleanup_state(state); cleanup_state(state)
