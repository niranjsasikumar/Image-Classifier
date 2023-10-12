from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor

def process_image(image_file: Any) -> Tensor:
    """Processes an image for prediction.

    Resizes and crops an image and converts the image to a Tensor that can be
    used with the model for predictions.

    Args:
        image_file: An image file object or the path to an image file.
    
    Returns:
        A Tensor that can be used with the model for predictions.
    """
    image = Image.open(image_file)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    
    width, height = image.size

    if width < height:
        new_size = (255, int(255 * height / width))
    else:
        new_size = (int(255 * width / height), 255)
    
    image = image.resize(new_size)
    width, height = image.size

    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))

    image = np.array(image)
    image = image.transpose((2, 0, 1))
    image = image / 255

    image[0] = (image[0] - 0.485) / 0.229
    image[1] = (image[1] - 0.456) / 0.224
    image[2] = (image[2] - 0.406) / 0.225

    image = image[np.newaxis,:]
    image = torch.from_numpy(image)
    image = image.float()

    return image