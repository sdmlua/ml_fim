import os
import argparse
import numpy as np
import pandas as pd
import rasterio
import joblib

def read_raster_with_meta(path):
    """Read a single-band raster and return its data array and nodata value."""
    print(f"Reading raster: {path}")
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
    return array, nodata

def preprocess_benchmark_folder(folder_path, date_str, output_dir):
    try:
        # File paths
        flood_path = os.path.join(folder_path, f"flood_depth_{date_str}.tif")
        owp_path = os.path.join(folder_path, f"owp_hand_fim{date_str}.tif")
        shared_rasters = {
            "aspect":    os.path.join(folder_path, f"aspect_{date_str}.tif"),
            "curvature": os.path.join(folder_path, f"curvature_{date_str}.tif"),
            "dem":       os.path.join(folder_path, f"dem_{date_str}.tif"),
            "lulc":      os.path.join(folder_path, f"lulc_{date_str}.tif"),
            "rem":       os.path.join(folder_path, f"rem_{date_str}.tif"),
        }

        # Handle slope file which might be either slope.tif or slope_<date>.tif
        slope_path = os.path.join(folder_path, f"slope_{date_str}.tif")
        if not os.path.exists(slope_path):
            slope_path = os.path.join(folder_path, "slope.tif")
        shared_rasters["slope"] = slope_path

        # Read base rasters
        owp_arr, owp_nodata = read_raster_with_meta(owp_path)
        flood_arr, flood_nodata = read_raster_with_meta(flood_path)

        if owp_arr.shape != flood_arr.shape:
            print(f"Shape mismatch: {owp_arr.shape} vs {flood_arr.shape}")
            return

        owp_flat = owp_arr.flatten()
        flood_flat = flood_arr.flatten()

        # Mask out OWP nodata
        mask = np.ones_like(owp_flat, dtype=bool)
        if owp_nodata is not None:
            mask &= (owp_flat != owp_nodata)

        # Impute flood nodata with 0
        if flood_nodata is not None:
            flood_flat[flood_flat == flood_nodata] = 0.0

        data = {
            "owp_hand_fim": owp_flat[mask],
            "flood_depth": flood_flat[mask],
        }

        # Process shared rasters
        for name, path in shared_rasters.items():
            arr, nodata = read_raster_with_meta(path)
            if arr.shape != owp_arr.shape:
                print(f"Skipping {name}: shape mismatch")
                return
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

        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.dropna(inplace=True)

        output_path = os.path.join(output_dir, f"data_{date_str}.pkl")
        joblib.dump(df, output_path)
        print(f"✅ Saved: {output_path} ({len(df)} rows)")

    except Exception as e:
        print(f"❌ Error processing {date_str}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to benchmark directory")
    parser.add_argument("--output", required=True, help="Output directory for pickle files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for subfolder in os.listdir(args.input):
        folder_path = os.path.join(args.input, subfolder)
        if os.path.isdir(folder_path):
            preprocess_benchmark_folder(folder_path, subfolder, args.output)

    print("\n✅ All done.")

if __name__ == "__main__":
    main()
