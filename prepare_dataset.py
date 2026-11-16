import os
import re
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
XRAYS_DIR = os.path.join(DATA_DIR, "xrays")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
CSVS_DIR = os.path.join(DATA_DIR, "CSVs")

os.makedirs(CSVS_DIR, exist_ok=True)


def normalize_filenames():
    # 1. Normalize xrays filenames to <number>.png
    for fname in os.listdir(XRAYS_DIR):
        old_path = os.path.join(XRAYS_DIR, fname)
        if not os.path.isfile(old_path):
            continue
        # Match <number>.png or <number>_png...
        match = re.match(r"^(\d+)_png\.rf\..*\.(jpg|png)$", fname, re.IGNORECASE)
        if match:
            num = match.group(1)
            new_fname = f"{num}.png"
            new_path = os.path.join(XRAYS_DIR, new_fname)
            with Image.open(old_path) as img:
                # Convert to grayscale 'L' to match the rest of the X-ray PNGs
                gray_img = img.convert("L")
                gray_img.save(new_path, format="PNG")
            if old_path != new_path:
                os.remove(old_path)
            print(f"Normalized xray: {fname} -> {new_fname}")

    # 2. Normalize masks filenames to <number>.png
    for fname in os.listdir(MASKS_DIR):
        old_path = os.path.join(MASKS_DIR, fname)
        if not os.path.isfile(old_path):
            continue
        match = re.match(r"^(\d+)_png\.rf\..*_mask\.png$", fname, re.IGNORECASE)
        if match:
            num = match.group(1)
            new_fname = f"{num}.png"
            new_path = os.path.join(MASKS_DIR, new_fname)
            if os.path.exists(new_path) and old_path != new_path:
                os.remove(new_path)
            os.rename(old_path, new_path)
            print(f"Normalized mask: {fname} -> {new_fname}")


def create_csvs():
    # Sort files numerically by their integer index
    xray_files = sorted(
        [f for f in os.listdir(XRAYS_DIR) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    mask_files_set = set(os.listdir(MASKS_DIR))

    rows = []
    for fname in xray_files:
        xray_rel_path = f"data/xrays/{fname}"
        if fname in mask_files_set:
            mask_rel_path = f"data/masks/{fname}"
        else:
            mask_rel_path = "None"
        rows.append({"xrays": xray_rel_path, "masks": mask_rel_path})

    dataset_df = pd.DataFrame(rows, columns=["xrays", "masks"])
    dataset_csv_path = os.path.join(CSVS_DIR, "dataset.csv")
    dataset_df.to_csv(dataset_csv_path, index=False)
    print(f"Saved {dataset_csv_path} with {len(dataset_df)} samples.")

    # Step 2: Create train / val / test CSVs using dataset.csv
    # Read back with keep_default_na=False so literal "None" is preserved as string "None"
    df = pd.read_csv(dataset_csv_path, keep_default_na=False)

    test_df = df[df["masks"] == "None"].reset_index(drop=True)
    labeled_df = df[df["masks"] != "None"].reset_index(drop=True)

    train_df, val_df = train_test_split(
        labeled_df, test_size=0.20, random_state=42, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_csv_path = os.path.join(CSVS_DIR, "train.csv")
    val_csv_path = os.path.join(CSVS_DIR, "val.csv")
    test_csv_path = os.path.join(CSVS_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Saved {train_csv_path} with {len(train_df)} samples.")
    print(f"Saved {val_csv_path} with {len(val_df)} samples.")
    print(f"Saved {test_csv_path} with {len(test_df)} samples.")

    return {
        "dataset.csv": len(dataset_df),
        "train.csv": len(train_df),
        "val.csv": len(val_df),
        "test.csv": len(test_df),
    }


def plot_verification_chart(counts):
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"], edgecolor="black", width=0.55)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("Number of Samples in Each CSV File", fontsize=14, fontweight="bold")
    plt.xlabel("CSV File", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.ylim(0, max(values) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    chart_path = os.path.join(BASE_DIR, "csv_samples_bar_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved bar chart to {chart_path}")


def create_submission_zip():
    zip_path = os.path.join(BASE_DIR, "csv_files_7.zip")
    csv_files = ["dataset.csv", "train.csv", "val.csv", "test.csv"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in csv_files:
            full_path = os.path.join(CSVS_DIR, fname)
            zf.write(full_path, arcname=fname)
    print(f"Created submission archive: {zip_path}")


if __name__ == "__main__":
    normalize_filenames()
    counts = create_csvs()
    plot_verification_chart(counts)
    create_submission_zip()

