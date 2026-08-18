import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import glob
from pathlib import Path

def load_nbaiot_data(data_dir):
    """Load N-BaIoT dataset from folder structure"""
    all_data = []
    device_summaries = []
    
    # Find all device folders
    device_folders = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    
    for device_path in device_folders:
        device_name = device_path.name
        print(f"\nProcessing device: {device_name}")
        
        # Find all CSV files in this device folder
        csv_files = list(device_path.glob("*.csv"))
        
        if not csv_files:
            print(f"  ⚠️ No CSV files found in {device_path}")
            continue
            
        device_data = []
        attack_classes = []
        
        for csv_file in csv_files:
            # Determine if it's benign or attack based on filename
            filename = csv_file.stem.lower()
            is_benign = 'benign' in filename or 'normal' in filename or 'benign' in csv_file.parent.name.lower()
            
            # Load the CSV
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                print(f"  Loaded {csv_file.name}: {len(df)} rows")
                
                # Add label
                label = 'Benign' if is_benign else 'Attack'
                if not is_benign:
                    attack_classes.append(filename)
                df['label'] = label
                df['attack_type'] = filename if not is_benign else 'Benign'
                df['device'] = device_name
                
                device_data.append(df)
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")
        
        if device_data:
            # Combine all data for this device
            combined = pd.concat(device_data, ignore_index=True)
            
            # Deduplicate
            before = len(combined)
            combined = combined.drop_duplicates()
            after = len(combined)
            
            # Counts
            benign_count = len(combined[combined['label'] == 'Benign'])
            attack_count = len(combined[combined['label'] == 'Attack'])
            n_attack_classes = len(set(combined[combined['label'] == 'Attack']['attack_type']))
            
            summary = {
                'device': device_name,
                'total_rows': after,
                'benign': benign_count,
                'attack': attack_count,
                'n_attack_classes': n_attack_classes,
                'attack_types': ', '.join(sorted(set(combined[combined['label'] == 'Attack']['attack_type'])))
            }
            device_summaries.append(summary)
            all_data.append(combined)
            
            print(f"  ✅ {device_name}: {benign_count} benign, {attack_count} attack ({n_attack_classes} classes)")
    
    if all_data:
        full_dataset = pd.concat(all_data, ignore_index=True)
        return full_dataset, pd.DataFrame(device_summaries)
    else:
        return None, pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to raw N-BaIoT data')
    parser.add_argument('--out-dir', default='data/processed', help='Output directory')
    args = parser.parse_args()
    
    print("📊 Loading N-BaIoT dataset...")
    dataset, summary = load_nbaiot_data(args.data_dir)
    
    if dataset is None:
        print("❌ No data found. Please check your data directory.")
        return
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(args.out_dir, 'preprocessing_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Split data
    print("\n📊 Splitting into train/test...")
    train, test = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['device'])
    
    print(f"\n📈 Final dataset:")
    print(f"  Total: {len(dataset)} rows")
    print(f"  Train: {len(train)} rows")
    print(f"  Test: {len(test)} rows")
    
    # Save train/test
    train_path = os.path.join(args.out_dir, 'train.csv')
    test_path = os.path.join(args.out_dir, 'test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"\n✅ Train saved to: {train_path}")
    print(f"✅ Test saved to: {test_path}")
    
    # Print summary table
    print("\n📊 Per-Device Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
