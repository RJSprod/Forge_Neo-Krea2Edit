from krea2edit_ref.patching import install_edit_forward
from krea2edit_ref.state import Krea2EditJobState, cleanup_state
class Model:
 def forward(self): return 1
class Engine:
 def get_learned_conditioning(self): return 1
def test_cleanup_is_idempotent():
 state=Krea2EditJobState(True, [], [], "fit", 768, 1, 1)
 cleanup_state(state); cleanup_state(state)


def test_instance_forward_wrapper_is_replaced_for_the_job_and_restored():
 class WrappedModel:
  patch = 2
  channels = 16
  class first:
   in_features = 64
  class last:
   class linear:
    out_features = 64
  def forward(self): return "class forward"

 model = WrappedModel()
 def forge_wrapper(*args, **kwargs): return "Forge wrapper"
 model.forward = forge_wrapper
 state = Krea2EditJobState(True, [], [], "fit", 768, 1, 1)

 install_edit_forward(model, Engine(), state)

 assert model.forward is not forge_wrapper
 cleanup_state(state)
 assert model.forward is forge_wrapper
