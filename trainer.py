import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from metrics import BCEDiceLoss, compute_dice_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Trainer:

    def __init__(self, model, train_loader, val_loader, args):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = args.device

        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
        )

        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=args.log_dir) if SummaryWriter is not None else None

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "lr": [],
        }

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        total_samples = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            bs = images.size(0)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            batch_dice = compute_dice_score(logits, masks, threshold=0.5, reduction="mean")
            running_loss += loss.item() * bs
            running_dice += batch_dice.item() * bs
            total_samples += bs

        epoch_loss = running_loss / max(total_samples, 1)
        epoch_dice = running_dice / max(total_samples, 1)
        return epoch_loss, epoch_dice

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_dices = []
        total_samples = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            bs = images.size(0)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            sample_dices = compute_dice_score(logits, masks, threshold=0.5, reduction="none")

            running_loss += loss.item() * bs
            all_dices.extend(sample_dices.cpu().numpy().tolist())
            total_samples += bs

        epoch_loss = running_loss / max(total_samples, 1)
        epoch_dice = float(np.mean(all_dices)) if all_dices else 0.0
        return epoch_loss, epoch_dice

    def fit(self):
        best_val_dice = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        print(
            f"Starting U-Net training on device='{self.device}' "
            f"for up to {self.args.epochs} epochs (bs={self.args.bs}, lr={self.args.lr}, wd={self.args.weight_decay})"
        )
        print("-" * 88)
        print(
            f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Dice':>11} | "
            f"{'Val Loss':>10} | {'Val Dice':>10} | {'LR':>10} | {'Status'}"
        )
        print("-" * 88)

        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_dice = self.train_one_epoch()
            val_loss, val_dice = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice"].append(train_dice)
            self.history["val_dice"].append(val_dice)
            self.history["lr"].append(current_lr)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Dice/train", train_dice, epoch)
                self.writer.add_scalar("Dice/val", val_dice, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            status = ""
            improved = False
            if val_dice > best_val_dice or (
                abs(val_dice - best_val_dice) < 1e-5 and val_loss < best_val_loss
            ):
                best_val_dice = val_dice
                best_val_loss = min(best_val_loss, val_loss)
                best_epoch = epoch
                epochs_without_improvement = 0
                improved = True
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_dice": best_val_dice,
                        "val_loss": val_loss,
                    },
                    self.args.best_model_path,
                )
                status = "* Saved best_unet.pth"
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                epochs_without_improvement += 1

            print(
                f"{epoch:6d} | {train_loss:11.4f} | {train_dice:11.4f} | "
                f"{val_loss:10.4f} | {val_dice:10.4f} | {current_lr:10.2e} | {status}"
            )

            if epochs_without_improvement >= self.args.early_stop_patience:
                print(
                    f"\nEarly stopping triggered at epoch {epoch} "
                    f"(no Val Dice improvement for {self.args.early_stop_patience} epochs)."
                )
                break

        elapsed = time.time() - start_time
        print("-" * 88)
        print(
            f"Training finished in {elapsed:.1f}s. Best Epoch: {best_epoch} | "
            f"Best Val Dice: {best_val_dice:.4f} | Best Model: {self.args.best_model_path}"
        )

        if self.writer is not None:
            self.writer.close()

        history_df = pd.DataFrame(self.history)
        history_csv = os.path.join(self.args.out_dir, "training_history.csv")
        history_df.to_csv(history_csv, index=False)
        self.plot_learning_curves()
        return best_val_dice, best_epoch

    def plot_learning_curves(self):
        epochs = self.history["epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Loss subplot
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        axes[0].plot(epochs, self.history["val_loss"], label="Val Loss", color="#d62728", linewidth=2)
        axes[0].set_title("Training & Validation Loss (BCE + Soft Dice)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        # Dice Score subplot
        axes[1].plot(epochs, self.history["train_dice"], label="Train Dice Score", color="#2ca02c", linewidth=2)
        axes[1].plot(epochs, self.history["val_dice"], label="Val Dice Score", color="#ff7f0e", linewidth=2)
        axes[1].set_title("Training & Validation Dice Score", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        plt.tight_layout()
        curve_path = os.path.join(self.args.out_dir, "learning_curves.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Saved learning curves to: {curve_path}")

