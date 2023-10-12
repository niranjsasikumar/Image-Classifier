from io import BytesIO

import requests
from torch import Tensor

from animalclassifier.image import process_image

class TestProcessImage:
    def test_local_image_path(self):
        image_path = "test_data/train/bear/1.jpg"
        image = process_image(image_path)
        assert type(image) is Tensor
    
    def test_web_image(self):
        response = requests.get(
            "https://images.unsplash.com/photo-1568162603664-fcd658421851"
        )
        image_file = BytesIO(response.content)
        image = process_image(image_file)
        assert type(image) is Tensor