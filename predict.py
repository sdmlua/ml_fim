import os
import argparse
import joblib
import numpy as np
import pandas as pd
from glob import glob

def load_model(model_path):
    return joblib.load(model_path)

def predict_and_save(model, path, log_target=False):
    df = joblib.load(path)
    if 'flood_depth' not in df.columns:
        return

    # Drop unused columns
    features = df.drop(columns=['flood_depth', 'huc8', 'return_period'], errors='ignore')
    features = features.select_dtypes(include=[np.number])

    preds = model.predict(features)

    if log_target:
        preds = np.expm1(preds)

    df['predicted'] = preds
    joblib.dump(df, path)
    print(f"✅ Updated: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model.pkl')
    parser.add_argument('--root_dir', required=True, help='Directory with *_test.pkl files')
    parser.add_argument('--log_target', action='store_true', help='If model was trained on log1p')
    args = parser.parse_args()

    model = load_model(args.model)
    test_files = glob(os.path.join(args.root_dir, "**", "*_test.pkl"), recursive=True)

    for path in test_files:
        try:
            predict_and_save(model, path, log_target=args.log_target)
        except Exception as e:
            print(f"❌ Failed on {path}: {e}")

if __name__ == "__main__":
    main()
