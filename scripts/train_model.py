import torch

from imageclassifier.dataloader import get_data_loader
from imageclassifier.model import get_model, train_model

model = get_model(40)
data_loader = get_data_loader("datasets/train", 10)
train_model(model, data_loader, 20)
torch.save(model, "models/model.pth")