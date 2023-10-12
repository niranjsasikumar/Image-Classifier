from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_data_loader(data_dir: str, batch_size: int) -> DataLoader:
    """Loads and preprocesses a dataset.

    Loads an image dataset and applies transformations to the images. Creates a
    data loader using the loaded dataset.

    Args:
        data_dir: Directory where the dataset is located.
        batch_size: Number of samples to load per batch.
    
    Returns:
        A data loader for the specified dataset that can be iterated through
        during training.
    """
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    dataset = ImageFolder(data_dir, transform=transformations)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader