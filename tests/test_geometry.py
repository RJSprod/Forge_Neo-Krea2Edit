from krea2edit_ref.geometry import reference_geometry
def test_fit_and_crop_geometry():
 assert reference_geometry(100,100,512,512).width == 512
 assert reference_geometry(100,300,512,512).width % 16 == 0
 assert reference_geometry(100,300,512,512).height % 16 == 0
 assert reference_geometry(100,300,512,512).offset_x >= 0
 assert reference_geometry(100,300,512,512,"crop (legacy)").width == 512
