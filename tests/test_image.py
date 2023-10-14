from PIL import Image
from torch import Tensor

from animalclassifier.image import get_image, process_image

class TestImage:
    class TestGetImage:
        def test_local_image(self):
            location = "test_data/train/bear/1.jpg"
            image = get_image(location)
            assert isinstance(image, Image.Image)
        
        def test_web_image(self):
            location = (
                "https://images.unsplash.com/photo-1568162603664-fcd658421851"
            )
            image = get_image(location)
            assert isinstance(image, Image.Image)
    
    class TestProcessImage:
        def test_valid_image(self):
            image = Image.open("test_data/train/bear/1.jpg")
            image = process_image(image)
            assert type(image) is Tensor