import os
import argparse
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
import joblib


def read_raster_with_meta(path):
    print(f"Reading raster: {path}")
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
    print(f"Finished reading: {path}")
    return array, nodata


def flatten_and_clean(arr, nodata, method="drop"):
    flat = arr.flatten()
    if nodata is not None:
        flat[flat == nodata] = np.nan
    if method == "drop":
        return flat
    elif method == "zero":
        return np.nan_to_num(flat, nan=0.0)
    elif method == "mean":
        mean_val = np.nanmean(flat)
        return np.nan_to_num(flat, nan=mean_val)
    else:
        raise ValueError(f"Invalid method: {method}")


def process_single_pair(owp_path, flood_path, shared_rasters, huc8, period_name, output_dir, test_size):
    print(f"\n--- Processing HUC8: {huc8}, Period: {period_name} ---")

    try:
        owp, owp_nodata = read_raster_with_meta(owp_path)
        flood, flood_nodata = read_raster_with_meta(flood_path)

        if owp.shape != flood.shape:
            print(f"Skipping due to shape mismatch: {owp.shape} vs {flood.shape}")
            return None

        df = pd.DataFrame()
        df["owp_hand_fim"] = flatten_and_clean(owp, owp_nodata, method="drop")
        df["flood_depth"] = flatten_and_clean(flood, flood_nodata, method="zero")

        for name, path in shared_rasters.items():
            print(f"Processing shared raster: {name}")
            arr, nodata = read_raster_with_meta(path)
            if arr.shape != owp.shape:
                print(f"Skipping due to shape mismatch in shared raster: {name}")
                return None
            if name == "aspect":
                radians = np.deg2rad(flatten_and_clean(arr, nodata, method="mean"))
                df["aspect_sin"] = np.sin(radians)
                df["aspect_cos"] = np.cos(radians)
            else:
                df[name] = flatten_and_clean(arr, nodata, method="mean")

        df["huc8"] = huc8
        df["return_period"] = period_name
        df.dropna(inplace=True)

        df["strata"] = pd.cut(df["flood_depth"], bins=10)

        train_df, test_df = train_test_split(
            df.drop(columns=["strata"]),
            test_size=test_size,
            stratify=df["strata"],
            random_state=42
        )

        test_dir = os.path.join(output_dir, huc8)
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, f"{period_name}_test.pkl")
        joblib.dump(test_df, test_file)
        print(f"Saved test set: {test_file}")
        print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
        return train_df

    except Exception as e:
        print(f"Error processing {huc8}/{period_name}: {repr(e)}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to base data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test split ratio (default 0.3)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Scanning input directory: {args.input}")

    tasks = []

    for huc8 in os.listdir(args.input):
        huc8_path = os.path.join(args.input, huc8)
        if not os.path.isdir(huc8_path):
            continue

        flood_dir = os.path.join(huc8_path, "flood_depth")
        owp_dir = os.path.join(huc8_path, "owp_hand_fim")

        if not os.path.isdir(flood_dir) or not os.path.isdir(owp_dir):
            continue

        shared_rasters = {
            "aspect": os.path.join(huc8_path, "aspect.tif"),
            "curvature": os.path.join(huc8_path, "curvature.tif"),
            "dem": os.path.join(huc8_path, "dem.tif"),
            "lulc": os.path.join(huc8_path, "lulc.tif"),
            "rem": os.path.join(huc8_path, "rem.tif"),
            "slope": os.path.join(huc8_path, "slope.tif"),
        }

        for owp_file in os.listdir(owp_dir):
            if not owp_file.endswith(".tif"):
                continue

            period_name = owp_file.replace(f"_{huc8}_depth.tif", "").replace(".tif", "")
            owp_path = os.path.join(owp_dir, owp_file)

            flood_path = None
            for fname in [
                f"fim_{period_name}_m_final.tif",
                f"{period_name}_{huc8}_depth.tif",
                f"fim_{period_name}.tif"
            ]:
                candidate = os.path.join(flood_dir, fname)
                if os.path.exists(candidate):
                    flood_path = candidate
                    break

            if flood_path:
                tasks.append((owp_path, flood_path, shared_rasters, huc8, period_name, args.output, args.test_size))
            else:
                print(f"Skipping: no flood file for {owp_file}")

    print(f"\nTotal raster pairs to process: {len(tasks)}\n")

    train_dfs = []
    for task in tasks:
        train_df = process_single_pair(*task)
        if train_df is not None:
            train_dfs.append(train_df)

    print("\nCombining all training data...")
    if not train_dfs:
        print("No training data generated.")
        return

    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    combined_train_df = combined_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_file = os.path.join(args.output, "train.pkl")
    joblib.dump(combined_train_df, train_file)
    print(f"Saved combined training file: {train_file}")
    print("Done.")


if __name__ == "__main__":
    main()
