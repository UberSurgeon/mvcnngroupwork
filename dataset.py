import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class SynchronizedSegmentationTransform:

    def __init__(
        self,
        img_size: int = 224,
        is_train: bool = False,
        crop_padding: int = 16,
        rotation_degrees: float = 10.0,
        brightness: float = 0.15,
        contrast: float = 0.15,
    ):
        self.img_size = img_size
        self.is_train = is_train

        if self.is_train:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.RandomCrop(size=img_size, padding=crop_padding, fill=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(
                    degrees=rotation_degrees,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ])

            self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.NEAREST,
                    antialias=False,
                ),
                transforms.RandomCrop(size=img_size, padding=crop_padding, fill=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(
                    degrees=rotation_degrees,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
            ])

            self.color_jitter = transforms.ColorJitter(
                brightness=brightness, contrast=contrast
            )
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ])
            self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.NEAREST,
                    antialias=False,
                ),
            ])
            self.color_jitter = None

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        seed = random.randint(0, 2**31 - 1)

        torch.manual_seed(seed)
        image_tensor = self.image_transform(image).float()

        if self.is_train and self.color_jitter is not None:
            image_tensor = torch.clamp(self.color_jitter(image_tensor), 0.0, 1.0)

        if mask is not None:
            torch.manual_seed(seed)
            mask_tensor = self.mask_transform(mask).float()
            mask_tensor = (mask_tensor > 0.5).float()
            return image_tensor, mask_tensor

        return image_tensor, None


class BoneSegmentationDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        base_dir: str,
        img_size: int = 224,
        is_train: bool = False,
        is_test: bool = False,
    ):
        super().__init__()
        self.csv_file = csv_file
        self.base_dir = base_dir
        self.img_size = img_size
        self.is_train = is_train
        self.is_test = is_test

        self.df = pd.read_csv(csv_file, keep_default_na=False)
        self.transform = SynchronizedSegmentationTransform(
            img_size=img_size, is_train=is_train
        )

    def _resolve_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path) and os.path.exists(rel_path):
            return rel_path
        candidate = os.path.join(self.base_dir, rel_path)
        if os.path.exists(candidate):
            return candidate
        return rel_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        xray_rel_path = str(row["xrays"]).strip()
        mask_rel_path = str(row.get("masks", "None")).strip()

        xray_full_path = self._resolve_path(xray_rel_path)
        xray_img = cv2.imread(xray_full_path, cv2.IMREAD_GRAYSCALE)
        if xray_img is None:
            raise FileNotFoundError(f"Could not read X-ray image at: {xray_full_path}")

        # Normalize pixel values to [0.0, 1.0] float32
        xray_float = xray_img.astype(np.float32) / 255.0

        has_valid_mask = (
            mask_rel_path != ""
            and mask_rel_path.lower() != "none"
            and mask_rel_path.lower() != "nan"
        )

        if not self.is_test:
            if not has_valid_mask:
                raise ValueError(
                    f"Missing mask for non-test sample at index {idx}: {xray_rel_path}"
                )
            mask_full_path = self._resolve_path(mask_rel_path)
            mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Could not read mask at: {mask_full_path}")

            # Binary float32 mask in {0.0, 1.0}
            mask_float = (mask_img > 0).astype(np.float32)
            x_tensor, y_tensor = self.transform(xray_float, mask_float)
            return {
                "image": x_tensor,
                "mask": y_tensor,
                "path": xray_rel_path,
            }
        else:
            if has_valid_mask:
                mask_full_path = self._resolve_path(mask_rel_path)
                mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_float = (mask_img > 0).astype(np.float32)
                    x_tensor, y_tensor = self.transform(xray_float, mask_float)
                    return {
                        "image": x_tensor,
                        "mask": y_tensor,
                        "path": xray_rel_path,
                    }
            x_tensor, _ = self.transform(xray_float, None)
            return {
                "image": x_tensor,
                "path": xray_rel_path,
            }


def get_dataloaders(args):
    train_dataset = BoneSegmentationDataset(
        csv_file=args.train_csv,
        base_dir=args.base_dir,
        img_size=args.img_size,
        is_train=True,
        is_test=False,
    )
    val_dataset = BoneSegmentationDataset(
        csv_file=args.val_csv,
        base_dir=args.base_dir,
        img_size=args.img_size,
        is_train=False,
        is_test=False,
    )
    test_dataset = BoneSegmentationDataset(
        csv_file=args.test_csv,
        base_dir=args.base_dir,
        img_size=args.img_size,
        is_train=False,
        is_test=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader
