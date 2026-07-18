import torch
from krea2edit_ref.forward import _imgids, _imgids_offset
def test_frame_ids_and_offsets():
 assert torch.all(_imgids(1,2,2,1)[...,0] == 1)
 ids = _imgids_offset(1,1,1,2,3,4)
 assert ids[0,0].tolist() == [2,3,4]
