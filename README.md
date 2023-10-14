# Image Classifier

A Python package for image classification. Built on the PyTorch framework, the imageclassifier package simplifies the process of creating and training an image classifier and making predictions using the classifier.

## Example

### Model Creation and Training

The `get_model` function returns a DenseNet model to use for classification tasks. The `get_data_loader` function returns a data loader that is used during training to sample images from the training dataset. The `train_model` function updates the weights of the classifier part of the model to the weights learned during training.

```python
from imageclassifier.dataloader import get_data_loader
from imageclassifier.model import get_model, train_model

num_labels = 40  # Number of labels in the classification task
model = get_model(num_labels)

data_dir = "./datasets/train"  # Training dataset directory
batch_size = 10
data_loader = get_data_loader(data_dir, batch_size)

epochs = 20
train_model(model, data_loader, epochs)
```

### Prediction using Model

The `get_image` function retrieves an image from a given URL or local file path. The `predict` function returns the index of the class with the highest probability and the associated probability.

```python
from imageclassifier.image import get_image
from imageclassifier.model import predict

image_location = "./datasets/test/test1.jpg"
image = get_image(image_location)

class_index, probability = predict(model, image)
```
