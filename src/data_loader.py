import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import glob
from pathlib import Path


def process_device(device_path, out_dir, test_size=0.2, seed=42):
    """Load, label, clean, split, and SAVE one device's data immediately.
    Returns a summary dict only (never holds the full dataset across devices)."""
    device_name = device_path.name
    print(f"\nProcessing device: {device_name}")

    csv_files = list(device_path.glob("*.csv"))
    if not csv_files:
        print(f"  [WARN] No CSV files found in {device_path}")
        return None

    device_data = []

    for csv_file in csv_files:
        filename = csv_file.stem.lower()
        is_benign = 'benign' in filename or 'normal' in filename or 'benign' in csv_file.parent.name.lower()

        try:
            df = pd.read_csv(csv_file, low_memory=False)
            print(f"  Loaded {csv_file.name}: {len(df)} rows")

            label = 'Benign' if is_benign else 'Attack'
            df['label'] = label
            df['attack_type'] = filename if not is_benign else 'Benign'
            df['device'] = device_name

            device_data.append(df)
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")

    if not device_data:
        return None

    combined = pd.concat(device_data, ignore_index=True)
    del device_data

    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    print(f"  Dropped {before - after} duplicate rows")

    benign_count = int((combined['label'] == 'Benign').sum())
    attack_count = int((combined['label'] == 'Attack').sum())
    n_attack_classes = combined.loc[combined['label'] == 'Attack', 'attack_type'].nunique()
    attack_types = ', '.join(sorted(combined.loc[combined['label'] == 'Attack', 'attack_type'].unique()))

    summary = {
        'device': device_name,
        'total_rows': after,
        'benign': benign_count,
        'attack': attack_count,
        'n_attack_classes': n_attack_classes,
        'attack_types': attack_types,
    }

    train, test = train_test_split(
        combined, test_size=test_size, random_state=seed, stratify=combined['label']
    )

    device_out = Path(out_dir) / device_name
    device_out.mkdir(parents=True, exist_ok=True)
    train.to_csv(device_out / 'train.csv', index=False)
    test.to_csv(device_out / 'test.csv', index=False)

    print(f"  [OK] {device_name}: {benign_count} benign, {attack_count} attack "
          f"({n_attack_classes} classes) -> train={len(train)} test={len(test)}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to raw N-BaIoT data')
    parser.add_argument('--out-dir', default='data/processed', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("[INFO] Loading N-BaIoT dataset...")
    os.makedirs(args.out_dir, exist_ok=True)

    device_folders = [d for d in Path(args.data_dir).iterdir() if d.is_dir()]
    if not device_folders:
        print("[ERROR] No device folders found. Please check your data directory.")
        return

    summaries = []
    for device_path in device_folders:
        summary = process_device(device_path, args.out_dir, args.test_size, args.seed)
        if summary:
            summaries.append(summary)

    if not summaries:
        print("[ERROR] No data was successfully processed.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, 'preprocessing_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[OK] Per-device train/test CSVs saved under: {args.out_dir}/<device_name>/")
    print(f"[OK] Summary saved to: {summary_path}")
    print("\n[INFO] Per-Device Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
