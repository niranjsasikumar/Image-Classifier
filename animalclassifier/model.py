from torchvision import models
from torch import nn

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