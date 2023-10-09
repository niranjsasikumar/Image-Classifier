import torchvision.models as models
import torch.nn as nn

def get_model(num_labels):
    """Returns a configured DenseNet model.

    Returns a DenseNet model that has its parameters frozen and its classifier
    set to a custom classifier composed of linear layers and ReLU activations.

    Args:
        num_labels: The number of labels in the classification task.
    """
    model = models.densenet161(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    
    classifier_num_inputs = model.classifier.in_features
    model.classifier = get_classifier(classifier_num_inputs, num_labels)

    return model

def get_classifier(num_inputs, num_outputs):
    """Returns a feedforward neural network.

    Returns a feedforward neural network consisting of fully connected linear
    layers and ReLU activation functions, with a final log-softmax activation
    function.

    Args:
        num_inputs: The number of input features.
        num_outputs: The number of output features.
    """
    classifier = nn.Sequential(
        nn.Linear(num_inputs, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, num_outputs),
        nn.LogSoftmax(dim=1)
    )

    return classifier