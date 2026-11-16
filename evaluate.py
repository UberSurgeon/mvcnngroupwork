import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from metrics import compute_dice_score


def save_augmentation_preview(train_dataset, out_dir: str, num_samples: int = 4):

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3.2 * num_samples))

    for i in range(num_samples):
        sample = train_dataset[i]
        img = sample["image"].squeeze(0).numpy()  # (H, W) in [0, 1]
        msk = sample["mask"].squeeze(0).numpy()   # (H, W) in {0, 1}
        path = sample["path"]

        axes[i, 0].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title(f"Augmented X-ray ({os.path.basename(path)})", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(msk, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("Synchronized Augmented Mask", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 2].imshow(np.ma.masked_where(msk < 0.5, msk), cmap="autumn", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Aligned Mask Overlay", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("Pre-Training Synchronized Data Augmentation Sanity Check", fontsize=13, fontweight="bold")
    plt.tight_layout()
    preview_path = os.path.join(out_dir, "augmentation_preview.png")
    plt.savefig(preview_path, dpi=150)
    plt.close()
    print(f"Saved augmentation sanity check preview to: {preview_path}")


@torch.no_grad()
def evaluate_validation_set(model, val_loader, args):
    """
    Evaluates the best saved checkpoint on the validation set, saves per-sample
    metrics to outputs/val_metrics.csv, and generates a validation prediction grid.
    """
    if os.path.exists(args.best_model_path):
        checkpoint = torch.load(args.best_model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"\nLoaded best checkpoint from {args.best_model_path} "
            f"(Epoch {checkpoint.get('epoch', '?')}, Val Dice: {checkpoint.get('val_dice', 0.0):.4f})"
        )

    model.eval()
    records = []
    vis_samples = []

    for batch in val_loader:
        images = batch["image"].to(args.device)
        masks = batch["mask"].to(args.device)
        paths = batch["path"]

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        sample_dices = compute_dice_score(logits, masks, threshold=0.5, reduction="none")

        for i in range(images.size(0)):
            dice_val = float(sample_dices[i].item())
            rel_path = paths[i]
            records.append({"xrays": rel_path, "dice_score": dice_val})
            if len(vis_samples) < 6:
                vis_samples.append(
                    {
                        "path": rel_path,
                        "image": images[i, 0].cpu().numpy(),
                        "gt": masks[i, 0].cpu().numpy(),
                        "pred": preds[i, 0].cpu().numpy(),
                        "dice": dice_val,
                    }
                )

    val_df = pd.DataFrame(records)
    val_metrics_path = os.path.join(args.out_dir, "val_metrics.csv")
    val_df.to_csv(val_metrics_path, index=False)

    mean_dice = float(val_df["dice_score"].mean())
    std_dice = float(val_df["dice_score"].std(ddof=1)) if len(val_df) > 1 else 0.0
    min_dice = float(val_df["dice_score"].min())
    max_dice = float(val_df["dice_score"].max())

    print("=" * 72)
    print(
        f"VALIDATION EVALUATION RESULTS ({len(val_df)} images):\n"
        f"  Mean Validation Dice Score : {mean_dice:.4f} ± {std_dice:.4f}\n"
        f"  Min / Max Validation Dice  : {min_dice:.4f} / {max_dice:.4f}\n"
        f"  Per-sample metrics saved to: {val_metrics_path}"
    )
    print("=" * 72)

    # Save validation visual comparison grid
    if vis_samples:
        n_rows = len(vis_samples)
        fig, axes = plt.subplots(n_rows, 4, figsize=(13, 3.0 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, item in enumerate(vis_samples):
            fname = os.path.basename(item["path"])
            img = item["image"]
            gt = item["gt"]
            pred = item["pred"]
            dice = item["dice"]

            axes[idx, 0].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 0].set_title(f"X-ray ({fname})", fontsize=10)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 1].set_title("Ground Truth Mask", fontsize=10)
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(pred, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 2].set_title(f"Predicted Mask (Dice: {dice:.4f})", fontsize=10)
            axes[idx, 2].axis("off")

            axes[idx, 3].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 3].imshow(
                np.ma.masked_where(pred < 0.5, pred), cmap="spring", alpha=0.45, vmin=0.0, vmax=1.0
            )
            axes[idx, 3].set_title("Prediction Overlay", fontsize=10)
            axes[idx, 3].axis("off")

        plt.suptitle(
            f"Validation Set Segmentation Results (Mean Dice: {mean_dice:.4f} ± {std_dice:.4f})",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        val_grid_path = os.path.join(args.out_dir, "val_predictions_grid.png")
        plt.savefig(val_grid_path, dpi=150)
        plt.close()
        print(f"Saved validation visual grid to: {val_grid_path}")

    return mean_dice, std_dice


@torch.no_grad()
def predict_test_set(model, test_loader, args):
    """
    Runs inference on all 50 test images in test_df.csv, saves predicted binary
    masks to outputs/test_predictions/, and generates side-by-side + semi-transparent
    overlay visualizations in outputs/test_overlays/.
    Also computes test Dice score automatically if ground-truth masks are present.
    """
    os.makedirs(args.test_pred_dir, exist_ok=True)
    os.makedirs(args.test_overlay_dir, exist_ok=True)

    model.eval()
    test_dices = []
    summary_samples = []

    for idx, batch in enumerate(test_loader):
        images = batch["image"].to(args.device)
        paths = batch["path"]

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        if "mask" in batch and batch["mask"] is not None:
            masks = batch["mask"].to(args.device)
            batch_dices = compute_dice_score(logits, masks, threshold=0.5, reduction="none")
            test_dices.extend(batch_dices.cpu().numpy().tolist())

        for i in range(images.size(0)):
            rel_path = paths[i]
            fname = os.path.basename(rel_path)
            img_np = images[i, 0].cpu().numpy()   # (H, W) in [0, 1]
            pred_np = preds[i, 0].cpu().numpy()   # (H, W) in {0, 1}
            prob_np = probs[i, 0].cpu().numpy()   # (H, W) in [0, 1]

            # Save binary mask image (0 and 255 uint8 PNG for clear visualization)
            pred_mask_uint8 = (pred_np * 255).astype(np.uint8)
            pred_save_path = os.path.join(args.test_pred_dir, fname)
            cv2.imwrite(pred_save_path, pred_mask_uint8)

            # Create side-by-side + semi-transparent overlay image using OpenCV (fast & crisp)
            img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.cvtColor(pred_mask_uint8, cv2.COLOR_GRAY2BGR)

            # Semi-transparent green/cyan tint on predicted bone region + contour outline
            overlay_bgr = img_bgr.copy()
            color_layer = np.zeros_like(img_bgr)
            color_layer[:, :, 1] = 255  # Green channel
            color_layer[:, :, 2] = 60   # Warm green-yellow tint
            fg_idx = pred_np > 0.5
            overlay_bgr[fg_idx] = cv2.addWeighted(
                img_bgr[fg_idx], 0.55, color_layer[fg_idx], 0.45, 0
            )
            contours, _ = cv2.findContours(
                pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 255), 1)

            # Combine horizontally: [Original X-ray | Predicted Binary Mask | Overlay]
            combined = np.hstack([img_bgr, mask_bgr, overlay_bgr])
            overlay_save_path = os.path.join(args.test_overlay_dir, fname)
            cv2.imwrite(overlay_save_path, combined)

            if len(summary_samples) < 8:
                summary_samples.append(
                    {
                        "fname": fname,
                        "image": img_np,
                        "pred": pred_np,
                        "prob": prob_np,
                    }
                )

    print(
        f"\nCompleted inference on {len(test_loader.dataset)} test images:\n"
        f"  Predicted binary masks saved to: {args.test_pred_dir}\n"
        f"  Side-by-side overlays saved to : {args.test_overlay_dir}"
    )

    if test_dices:
        mean_test_dice = float(np.mean(test_dices))
        std_test_dice = float(np.std(test_dices, ddof=1)) if len(test_dices) > 1 else 0.0
        print(f"  Ground-Truth Test Dice Score   : {mean_test_dice:.4f} ± {std_test_dice:.4f}")

    # Generate a summary grid of test predictions & overlays
    if summary_samples:
        n_rows = len(summary_samples)
        fig, axes = plt.subplots(n_rows, 3, figsize=(10.5, 2.8 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, item in enumerate(summary_samples):
            fname = item["fname"]
            img = item["image"]
            pred = item["pred"]

            axes[idx, 0].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 0].set_title(f"Test X-ray ({fname})", fontsize=10)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(pred, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 1].set_title("Predicted Binary Mask", fontsize=10)
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 2].imshow(
                np.ma.masked_where(pred < 0.5, pred), cmap="Wistia", alpha=0.45, vmin=0.0, vmax=1.0
            )
            axes[idx, 2].set_title("Predicted Mask Overlay", fontsize=10)
            axes[idx, 2].axis("off")

        plt.suptitle(
            "Unlabeled Test Set Predictions & Overlays (Sample Preview)",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        summary_path = os.path.join(args.out_dir, "test_overlays_summary.png")
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"  Test summary figure saved to   : {summary_path}")


if __name__ == "__main__":
    from args import get_args
    from dataset import get_dataloaders
    from model import UNet

    args = get_args()
    _, val_loader, test_loader = get_dataloaders(args)
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        dropout_p=args.dropout,
    ).to(args.device)
    evaluate_validation_set(model, val_loader, args)
    predict_test_set(model, test_loader, args)


