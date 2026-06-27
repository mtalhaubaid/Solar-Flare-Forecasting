"""PyTorch datasets for solar active-region image classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config
from .preprocessing import add_binary_labels, attach_image_paths


def build_transforms(image_size: int = config.IMAGE_SIZE, train: bool = False):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class SolarFlareImageDataset(Dataset):
    """Load image patches and binary flare labels from a CSV file."""

    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame,
        image_root: str | Path | None = config.SDOBENCHMARK_DIR,
        transform=None,
        image_col: str | None = None,
        path_col: str = config.IMAGE_PATH_COLUMN,
        label_col: str = config.LABEL_COLUMN,
        peak_flux_col: str | None = None,
        channel: str | None = "hmi",
        strict: bool = True,
        return_path: bool = False,
    ) -> None:
        if isinstance(csv_path, pd.DataFrame):
            frame = csv_path.copy()
        else:
            frame = pd.read_csv(csv_path)

        if label_col not in frame.columns:
            frame = add_binary_labels(frame, peak_flux_col=peak_flux_col, label_col=label_col)

        if path_col not in frame.columns:
            frame = attach_image_paths(frame, image_root=image_root, image_col=image_col, channel=channel, output_col=path_col)

        if path_col not in frame.columns:
            raise ValueError(
                "No image path column found. Provide image_path in the CSV or pass --image-col "
                "with a metadata column that points to images."
            )

        frame[path_col] = frame[path_col].astype(str)
        if strict:
            missing = [path for path in frame[path_col] if not Path(path).exists()]
            if missing:
                examples = ", ".join(missing[:5])
                raise FileNotFoundError(f"{len(missing)} image files are missing. Examples: {examples}")

        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row[self.path_col])
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        if self.return_path:
            return image, label, str(path)
        return image, label


def make_dataloader(
    dataset: Dataset,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

