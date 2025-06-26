import argparse
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .pkl file")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input}")
    df = joblib.load(args.input)

    print(f"Original dataset shape: {df.shape}")

    # Ensure the target column is included and first
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in dataset.")
        return

    # Reorder columns to put target first, then others
    cols = [args.target] + [col for col in df.columns if col != args.target]
    df = df[cols]

    # Keep only numeric columns (including target)
    df = df.select_dtypes(include=['number'])

    print("\nDescriptive statistics (including target):\n")
    print(df.describe())

if __name__ == "__main__":
    main()
