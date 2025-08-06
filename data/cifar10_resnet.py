import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size: int, num_workers: int, data_root: str = "data/"):
    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),               # [0,1]
        transforms.Lambda(lambda x: 2*x-1),  # [−1,1]
    ])
    train_ds = datasets.CIFAR10(data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=tf)
    return (
        DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
