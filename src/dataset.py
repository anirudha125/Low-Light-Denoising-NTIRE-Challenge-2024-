import os
import json
from typing import List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class LowLightDataset(Dataset):
    """
    Dataset loader for paired low-light enhancement images.
    Preserves native resolution without unwanted resizing.
    """
    def __init__(self, dataset_root: str, file_list: List[str], transform=None):
        self.dataset_root = dataset_root
        self.low_dir = os.path.join(dataset_root, "low")
        self.high_dir = os.path.join(dataset_root, "high")
        self.file_list = sorted(file_list)
        self.transform = transform if transform is not None else transforms.ToTensor()

        # Validate that all files exist in both low and high dirs
        for f in self.file_list:
            low_p = os.path.join(self.low_dir, f)
            high_p = os.path.join(self.high_dir, f)
            if not os.path.isfile(low_p):
                raise FileNotFoundError(f"Missing low image: {low_p}")
            if not os.path.isfile(high_p):
                raise FileNotFoundError(f"Missing high image: {high_p}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.file_list[idx]
        low_path = os.path.join(self.low_dir, fname)
        high_path = os.path.join(self.high_dir, fname)

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        return low_tensor, high_tensor, fname


def get_dataloaders(
    dataset_root: str,
    split_json_path: str,
    batch_size: int = 4,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    Loads train and val datasets using the fixed, persisted split.
    """
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_files = split_data["train_files"]
    val_files = split_data["val_files"]

    train_dataset = LowLightDataset(dataset_root, train_files, transform=transforms.ToTensor())
    val_dataset = LowLightDataset(dataset_root, val_files, transform=transforms.ToTensor())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # batch_size=1 for evaluation to evaluate each image independently
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_files, val_files
