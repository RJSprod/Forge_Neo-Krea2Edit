import torch
from krea2edit_ref.forward import build_reference_bias
def test_bias_only_changes_target_reference_block():
 assert build_reference_bias(1, 2, [3], 4, [1], "cpu", torch.float32) is None
 bias = build_reference_bias(1, 2, [3, 2], 4, [2,.5], "cpu", torch.float32)
 assert bias[0,0,5:,2:5].gt(0).all() and bias[0,0,5:,5:7].lt(0).all()
 assert bias[0,0,:5].eq(0).all()
