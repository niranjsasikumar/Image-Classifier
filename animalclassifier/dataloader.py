from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_data_loader(
    data_dir: str,
    transformations: Compose,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Loads and preprocesses a dataset.

    Loads an image dataset and applies transformations to the images. Creates a
    data loader using the loaded dataset.

    Args:
        data_dir: Directory where the dataset is located.
        transformations: Transformations to apply to the images in the dataset.
        batch_size: Number of samples to load per batch.
        shuffle: Whether the data samples should be reshuffled in each epoch.
    
    Returns:
        A data loader for the specified dataset that can be iterated through
        during training.
    """
    dataset = ImageFolder(data_dir, transform=transformations)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader