import os
import argparse
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
import joblib

def read_raster_with_meta(path):
    """Read a single-band raster and return its data array and nodata value."""
    print(f"Reading raster: {path}")
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
    print(f"Finished reading: {path}")
    return array, nodata

def process_single_pair(owp_path, flood_path, shared_rasters,
                        huc8, period_name, output_dir, test_size):
    """
    Read OWP + flood rasters, mask/impute nodata,
    balance zeros vs non-zeros, build DataFrame,
    drop tiny strata, stratify & split, and save the test set.
    """
    print(f"\n--- Processing HUC8: {huc8}, Period: {period_name} ---")
    try:
        # 1) Read base rasters
        owp_arr, owp_nodata     = read_raster_with_meta(owp_path)
        flood_arr, flood_nodata = read_raster_with_meta(flood_path)

        # 2) Ensure shapes match
        if owp_arr.shape != flood_arr.shape:
            print(f"Skipping due to shape mismatch: {owp_arr.shape} vs {flood_arr.shape}")
            return None

        # 3) Flatten base rasters
        owp_flat   = owp_arr.flatten()
        flood_flat = flood_arr.flatten()

        # 4) Mask out owp nodata
        mask = np.ones_like(owp_flat, dtype=bool)
        if owp_nodata is not None:
            mask &= (owp_flat != owp_nodata)

        # 5) Impute flood nodata → 0 then apply mask
        if flood_nodata is not None:
            flood_flat[flood_flat == flood_nodata] = 0.0
        owp_clean   = owp_flat[mask]
        flood_clean = flood_flat[mask]

        # 6) Initialize data dict
        data = {
            "owp_hand_fim": owp_clean,
            "flood_depth":  flood_clean,
        }

        # 7) Process shared rasters: mask, impute nodata→mean
        for name, path in shared_rasters.items():
            arr, nodata = read_raster_with_meta(path)
            if arr.shape != owp_arr.shape:
                print(f"Skipping due to shape mismatch in shared raster: {name}")
                return None

            arr_flat = arr.flatten().astype(np.float32)
            if nodata is not None:
                arr_flat[arr_flat == nodata] = np.nan
            mean_val = np.nanmean(arr_flat)
            arr_flat[np.isnan(arr_flat)] = mean_val

            arr_clean = arr_flat[mask]
            if name == "aspect":
                radians = np.deg2rad(arr_clean)
                data["aspect_sin"] = np.sin(radians)
                data["aspect_cos"] = np.cos(radians)
            else:
                data[name] = arr_clean

        # 8) Build DataFrame, add metadata, drop any NaNs
        df = pd.DataFrame(data)
        df["huc8"]          = huc8
        df["return_period"] = period_name
        df.dropna(inplace=True)

        # 9) Balance zeros vs non-zeros in flood_depth
        zeros    = df[df["flood_depth"] == 0]
        nonzeros = df[df["flood_depth"] > 0]
        if len(zeros) and len(nonzeros):
            n = min(len(zeros), len(nonzeros))
            zeros    = zeros.sample(n=n, random_state=42)
            nonzeros = nonzeros.sample(n=n, random_state=42)
            df = pd.concat([zeros, nonzeros], ignore_index=True)
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # 10) Create equal-width strata and drop any with fewer than 2 samples
        df["strata"] = pd.cut(df["flood_depth"], bins=10)
        counts = df["strata"].value_counts()
        valid = counts[counts >= 2].index
        df = df[df["strata"].isin(valid)].reset_index(drop=True)

        # 11) Stratified train/test split
        train_df, test_df = train_test_split(
            df.drop(columns=["strata"]),
            test_size=test_size,
            stratify=df["strata"],
            random_state=42
        )

        # 12) Save test set
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
    parser.add_argument("--input",  type=str, required=True,
                        help="Base data directory with HUC8 subfolders")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for train.pkl and test sets")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Fraction of data for test set (default 0.3)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Scanning input directory: {args.input}")

    tasks = []
    for huc8 in os.listdir(args.input):
        huc8_path = os.path.join(args.input, huc8)
        if not os.path.isdir(huc8_path):
            continue

        flood_dir = os.path.join(huc8_path, "flood_depth")
        owp_dir   = os.path.join(huc8_path, "owp_hand_fim")
        if not (os.path.isdir(flood_dir) and os.path.isdir(owp_dir)):
            continue

        shared_rasters = {
            "aspect":    os.path.join(huc8_path, "aspect.tif"),
            "curvature": os.path.join(huc8_path, "curvature.tif"),
            "dem":       os.path.join(huc8_path, "dem.tif"),
            "lulc":      os.path.join(huc8_path, "lulc.tif"),
            "rem":       os.path.join(huc8_path, "rem.tif"),
            "slope":     os.path.join(huc8_path, "slope.tif"),
        }

        for owp_file in os.listdir(owp_dir):
            if not owp_file.endswith(".tif"):
                continue
            period = owp_file.replace(f"_{huc8}_depth.tif", "").replace(".tif", "")
            owp_p  = os.path.join(owp_dir, owp_file)

            flood_p = None
            for fname in (f"fim_{period}_m_final.tif",
                          f"{period}_{huc8}_depth.tif",
                          f"fim_{period}.tif"):
                candidate = os.path.join(flood_dir, fname)
                if os.path.exists(candidate):
                    flood_p = candidate
                    break

            if flood_p:
                tasks.append((owp_p, flood_p, shared_rasters,
                              huc8, period, args.output, args.test_size))
            else:
                print(f"Skipping {owp_file}: no matching flood file found")

    print(f"\nTotal raster pairs to process: {len(tasks)}\n")

    train_dfs = []
    for task in tasks:
        df_train = process_single_pair(*task)
        if df_train is not None:
            train_dfs.append(df_train)

    if train_dfs:
        combined = pd.concat(train_dfs, ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_train = os.path.join(args.output, "train.pkl")
        joblib.dump(combined, out_train)
        print(f"\nSaved combined training file: {out_train}")
    else:
        print("No training data generated.")

    print("Done.")

if __name__ == "__main__":
    main()
