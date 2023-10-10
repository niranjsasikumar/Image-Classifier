import pytest
from animalclassifier import model

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