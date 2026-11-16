import os
import argparse
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args(cli_args=None):
    parser = argparse.ArgumentParser(
        description="PyTorch 2D U-Net Bone Segmentation Pipeline"
    )

    # Paths and directories
    parser.add_argument(
        "--base_dir",
        type=str,
        default=BASE_DIR,
        help="Project root directory containing data/",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=os.path.join(BASE_DIR, "data", "CSVs"),
        help="Directory containing dataset CSV files",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join(BASE_DIR, "data", "CSVs", "train_df.csv"),
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=os.path.join(BASE_DIR, "data", "CSVs", "val_df.csv"),
        help="Path to validation CSV file",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=os.path.join(BASE_DIR, "data", "CSVs", "test_df.csv"),
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(BASE_DIR, "outputs"),
        help="Output directory for checkpoints, plots, and predictions",
    )

    # Training hyperparameters
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs")
    parser.add_argument("--img_size", type=int, default=224, help="Input image height/width")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout2d probability in deeper U-Net blocks",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Base channel count for U-Net first stage",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau scheduler",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=15,
        help="Patience for early stopping",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )

    args = parser.parse_args(args=cli_args)

    # Derived output paths
    args.best_model_path = os.path.join(args.out_dir, "best_unet.pth")
    args.test_pred_dir = os.path.join(args.out_dir, "test_predictions")
    args.test_overlay_dir = os.path.join(args.out_dir, "test_overlays")
    args.log_dir = os.path.join(args.out_dir, "logs")

    return args

