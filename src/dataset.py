import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NORMAL_CLASS = 0  # رقم 0 به عنوان کلاس "نرمال" در نظر گرفته می‌شود

transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_train_loader(batch_size=128):
    """فقط نمونه‌های کلاس نرمال (رقم 0) برای آموزش autoencoder"""
    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    normal_idx = [i for i, label in enumerate(train_set.targets) if label == NORMAL_CLASS]
    normal_subset = Subset(train_set, normal_idx)
    return DataLoader(normal_subset, batch_size=batch_size, shuffle=True, num_workers=2)


def get_test_loader(batch_size=128):
    """داده تست شامل نرمال (0) و anomaly (سایر ارقام) برای ارزیابی"""
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2), test_set.targets
