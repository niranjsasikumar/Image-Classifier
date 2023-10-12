import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models

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