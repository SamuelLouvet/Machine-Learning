import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

try:
    from torchsampler import ImbalancedDatasetSampler
except Exception:
    try:
        from .torchsampler import ImbalancedDatasetSampler
    except Exception:
        ImbalancedDatasetSampler = None


def build_transforms(image_size: int = 32, mean: float = 0.5, std: float = 0.5, augment: bool = False):
    ops = [transforms.Grayscale(), transforms.Resize((image_size, image_size))]
    if augment:
        # Mild augmentations to help generalization and balance
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))]
    return transforms.Compose(ops)


def create_datasets(train_dir: str, test_dir: str, transform=None):
    if transform is None:
        transform = build_transforms()
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    return train_data, test_data


def split_train_valid_indices(num_train: int, valid_size: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    indices = list(range(num_train))
    rng.shuffle(indices)
    split_tv = int(np.floor(valid_size * num_train))
    valid_idx = indices[:split_tv]
    train_idx = indices[split_tv:]
    return train_idx, valid_idx


def create_loaders(
    train_data,
    test_data,
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = 2,
    use_imbalanced_sampler: bool = False,
    seed: int = 42,
):
    train_idx, valid_idx = split_train_valid_indices(len(train_data), valid_size, seed)
    train_subset = Subset(train_data, train_idx)
    valid_subset = Subset(train_data, valid_idx)

    if use_imbalanced_sampler and ImbalancedDatasetSampler is not None:
        train_sampler = ImbalancedDatasetSampler(train_subset)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def create_data(train_dir: str = './train_images', test_dir: str = './test_images', batch_size: int = 32,
                valid_size: float = 0.2, num_workers: int = 2, image_size: int = 32,
                normalize_mean: float = 0.5, normalize_std: float = 0.5, use_imbalanced_sampler: bool = False,
                seed: int = 42, augment: bool = False):
    transform = build_transforms(image_size=image_size, mean=normalize_mean, std=normalize_std, augment=augment)
    train_data, test_data = create_datasets(train_dir, test_dir, transform)
    train_loader, valid_loader, test_loader = create_loaders(
        train_data, test_data, batch_size=batch_size, valid_size=valid_size,
        num_workers=num_workers, use_imbalanced_sampler=use_imbalanced_sampler, seed=seed
    )
    classes = tuple(train_data.classes)
    return train_loader, valid_loader, test_loader, classes

