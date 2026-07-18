from PIL import Image
from krea2edit_ref.image_prep import extract_gallery
def test_gallery_normalizes_rgba_and_limits_count():
 image = Image.new("RGBA", (2,2), (1,2,3,127))
 assert extract_gallery([(image,"x")])[0].mode == "RGB"
 try: extract_gallery([image,image,image])
 except RuntimeError: pass
 else: assert False
