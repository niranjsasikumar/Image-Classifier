from PIL.Image import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models

from imageclassifier.image import process_image

def get_model(num_labels: int) -> models.DenseNet:
    """Returns a configured DenseNet model.

    Returns a DenseNet model that has its parameters frozen and its classifier
    set to a custom classifier composed of linear layers and ReLU activations.

    Args:
        num_labels: Number of labels in the classification task.
    """
    if num_labels > 512:
        raise ValueError("num_labels must be less than or equal to 512")
    
    pretrained_weights = models.DenseNet161_Weights.DEFAULT
    model = models.densenet161(weights=pretrained_weights)

    for parameter in model.parameters():
        parameter.requires_grad = False
    
    num_inputs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_inputs, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, num_labels),
        nn.LogSoftmax(dim=1)
    )

    return model

def train_model(
    model: models.DenseNet, data_loader: DataLoader, epochs: int
) -> None:
    """Trains the classifier of a model.

    Trains the classifier of a DenseNet model using the negative log-likelihood
    loss function and Adam optimization.

    Args:
        model: Model to be trained.
        data_loader: Data loader used to train the model.
        epochs: Number of epochs during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.classifier.parameters())
    criterion = nn.NLLLoss()
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

def predict(model: models.DenseNet, image: Image) -> tuple[int, float]:
    """Predicts the class and associated probability for an image.

    Performs a forward pass through a model using an input image, computes class
    probabilities, and returns the predicted class index and its associated
    probability.

    Args:
        model: DenseNet model to use for prediction.
        image: Image to classify, passed as input to the model.
    
    Returns:
        A tuple (class_index, probability) where class_index is the index of the
        predicted class and probability is the associated probability.
    """
    image_tensor = process_image(image)
    model.eval()
    output = model.forward(image_tensor)
    output = torch.exp(output)
    probability, class_index = output.topk(1, dim=1)
    probability = probability.item()
    class_index = class_index.item()
    return (class_index, probability)