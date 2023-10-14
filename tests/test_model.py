import copy

from PIL import Image
import pytest
import torch

from animalclassifier import model
from animalclassifier.dataloader import get_data_loader

class TestModel:
    class TestGetModel:
        def test_valid_num_labels(self):
            num_labels = 40
            test_model = model.get_model(num_labels)
            parameter = next(test_model.parameters())
            assert parameter.requires_grad == False
            out_features = test_model.classifier.pop(4).out_features
            assert out_features == num_labels
        
        def test_invalid_num_labels(self):
            num_labels = 600
            with pytest.raises(ValueError) as exception_info:
                test_model = model.get_model(num_labels)
            assert "less than or equal to" in str(exception_info.value)
    
    class TestTrainModel:
        def test_valid_arguments(self):
            initial_model = model.get_model(2)
            trained_model = copy.deepcopy(initial_model)
            data_loader = get_data_loader("test_data/train", 10)
            epochs = 1
            model.train_model(trained_model, data_loader, epochs)
            
            for initial_parameter, trained_parameter in zip(
                initial_model.classifier.parameters(),
                trained_model.classifier.parameters()
            ):
                assert not torch.equal(initial_parameter, trained_parameter)
    
    class TestPredict:
        def test_valid_arguments(self):
            test_model = torch.load("test_data/models/test_model.pth")
            image = Image.open("test_data/train/bear/1.jpg")
            animal, probability = model.predict(test_model, image)
            assert animal == 0
            assert probability > 0.5