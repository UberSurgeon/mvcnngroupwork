import os
import shutil
import random
import numpy as np
import torch

from args import get_args
from dataset import get_dataloaders
from model import UNet
from trainer import Trainer
from evaluate import (
    save_augmentation_preview,
    evaluate_validation_set,
    predict_test_set,
)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_csv_aliases(csv_dir: str):
    """
    Ensures both (train.csv, val.csv, test.csv) and
    (train_df.csv, val_df.csv, test_df.csv) exist in data/CSVs/
    without overwriting existing files.
    """
    pairs = [
        ("train.csv", "train_df.csv"),
        ("val.csv", "val_df.csv"),
        ("test.csv", "test_df.csv"),
    ]
    for src_name, dst_name in pairs:
        src_path = os.path.join(csv_dir, src_name)
        dst_path = os.path.join(csv_dir, dst_name)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.exists(dst_path) and not os.path.exists(src_path):
            shutil.copy2(dst_path, src_path)


def main():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ensure_csv_aliases(args.csv_dir)

    # Build datasets and dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print(
        f"Loaded datasets -> Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )

    save_augmentation_preview(train_loader.dataset, args.out_dir, num_samples=4)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        dropout_p=args.dropout,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized 2D U-Net with {total_params:,} trainable parameters on {args.device}.")

    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_out = model(sample_batch["image"].to(args.device))
    print(
        f"Shape check -> Input batch: {tuple(sample_batch['image'].shape)}, "
        f"Mask batch: {tuple(sample_batch['mask'].shape)}, "
        f"Logits batch: {tuple(sample_out.shape)}\n"
    )

    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.fit()

    evaluate_validation_set(model, val_loader, args)

    predict_test_set(model, test_loader, args)


if __name__ == "__main__":
    main()

