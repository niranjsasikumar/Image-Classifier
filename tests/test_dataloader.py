from torchvision import transforms
from animalclassifier.dataloader import get_data_loader
from torch.utils.data import DataLoader

class TestGetDataLoader:
    def test_valid_arguments(self):
        data_dir = "./test_data/train"
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
        batch_size = 10
        shuffle = True

        data_loader = get_data_loader(
            data_dir, transformations, batch_size, shuffle
        )

        assert type(data_loader) is DataLoader