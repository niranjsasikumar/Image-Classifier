from torch.utils.data import DataLoader

from imageclassifier.dataloader import get_data_loader

class TestGetDataLoader:
    def test_valid_arguments(self):
        data_dir = "test_data/train"
        batch_size = 10
        data_loader = get_data_loader(data_dir, batch_size)
        assert type(data_loader) is DataLoader