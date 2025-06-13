import os
import toml
import argparse
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from imblearn.over_sampling import SMOTE

def log(msg: str):
    print(f"[ml_fim] {msg}")

def load_raster_as_array(path: Path) -> tuple[np.ndarray, float | None]:
    log(f"Attempting to read raster: {path}")
    try:
        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float32)
            nodata = src.nodata
        return array, nodata
    except rasterio.errors.RasterioIOError as e:
        log(f"Error reading raster {path}: {e}")
        raise

def create_dataframe(feature_paths: dict[str, Path]) -> pd.DataFrame:
    log("Creating DataFrame from rasters…")
    arrays = []
    names = []

    for name, path in feature_paths.items():
        try:
            arr, nodata = load_raster_as_array(path)
            if nodata is not None:
                log(f"Replacing nodata for '{name}' ({nodata}) with NaN")
                arr[np.isclose(arr, nodata)] = np.nan
            arrays.append(arr)
            names.append(name)
        except Exception as e:
            log(f"WARNING: Skipping feature '{name}' due to error: {e}")
            continue

    if not arrays:
        log("No valid raster arrays were loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    stacked = np.stack(arrays, axis=-1)
    flat = stacked.reshape(-1, stacked.shape[-1]) 
    return pd.DataFrame(flat, columns=names)

def get_next_output_id(base_path: Path) -> int:
    base_path.mkdir(parents=True, exist_ok=True)
    existing_dirs = [p.name for p in base_path.iterdir() if p.is_dir() and p.name.startswith("data_")]
    max_id = 0
    for dir_name in existing_dirs:
        try:
            num = int(dir_name.replace("data_", ""))
            max_id = max(max_id, num)
        except ValueError:
            log(f"Warning: Non-standard directory name '{dir_name}' in {base_path}.")
    return max_id + 1

def main(toml_path: str, classification: bool = False, smote: bool = False, name: str = None):
    log("Starting MLFIM Preprocessing Workflow…")
    try:
        log(f"Loading configuration from: {toml_path}")
        cfg = toml.load(toml_path)
    except FileNotFoundError:
        log(f"ERROR: Configuration file not found at '{toml_path}'.")
        return
    except toml.TomlDecodeError as e:
        log(f"ERROR: Failed to parse TOML file: {e}.")
        return

    project_root = Path(".").resolve()
    base_data_id = str(cfg.get("base_path", "")).replace("/data/raw/", "").replace("data/raw/", "").strip("/")
    if not base_data_id:
        log("ERROR: 'base_path' is missing or invalid in the TOML file.")
        return

    log(f"Cleaned base path: {base_data_id}")

    features_cfg = cfg.get("features", {})
    if not features_cfg:
        log("WARNING: No features defined in [features].")
        return

    feature_paths = {
        k: project_root / "data" / "raw" / base_data_id / v for k, v in features_cfg.items()
    }
    for k, path in feature_paths.items():
        log(f"Feature '{k}': {path}")

    df = create_dataframe(feature_paths)
    if df.empty:
        log("DataFrame is empty. Exiting.")
        return

    if classification:
        target = cfg.get("target")
        threshold = cfg.get("classification_threshold", 0.1)
        class_target = cfg.get("classification_target", "flooded")
        if target not in df.columns:
            log(f"ERROR: Target column '{target}' not found in DataFrame.")
            return
        df[class_target] = (df[target] > threshold).astype(int)
        log(f"Classification column '{class_target}' added with threshold {threshold}")

        if smote:
            log("Applying SMOTE to classification data…")
            from sklearn.impute import SimpleImputer
            features = df.drop(columns=[target, class_target])
            labels = df[class_target]
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(features)
            y = labels.values
            sm = SMOTE()
            X_resampled, y_resampled = sm.fit_resample(X, y)
            df = pd.DataFrame(X_resampled, columns=features.columns)
            df[class_target] = y_resampled
            log("SMOTE applied successfully.")

    out_dir = project_root / "data" / "processed" / base_data_id

    if name:
        output_path = out_dir / name
        if output_path.exists():
            log(f"ERROR: Directory '{name}' already exists under {out_dir}.")
            return
    else:
        output_id = get_next_output_id(out_dir)
        output_path = out_dir / f"data_{output_id}"

    output_path.mkdir(parents=True, exist_ok=False)

    df.to_pickle(output_path / "full_data.pkl")
    log(f"Saved DataFrame to {output_path / 'full_data.pkl'}")

    with open(output_path / "info.txt", "w") as f:
        f.write("MLFIM Data Preprocessing Summary\n" + "=" * 40 + "\n\n")
        f.write(f"Processed Base ID: {base_data_id}\n")
        f.write(f"Project Root: {project_root}\n")
        f.write("Features:\n" + "-" * 25 + "\n")
        for k, path in feature_paths.items():
            f.write(f"{k:15s}: {path}\n")
        f.write("\nData Statistics:\n" + "-" * 40 + "\n")
        f.write(df.describe().to_string())
        f.write("\n")

    log("Preprocessing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLFIM Preprocessing")
    parser.add_argument("--config", type=str, default="config/inputs.toml", help="Path to config TOML")
    parser.add_argument("--classification", action="store_true", help="Generate classification labels")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE (only with --classification)")
    parser.add_argument("--name", type=str, help="Name of output directory (must be unique)")
    args = parser.parse_args()

    if args.smote and not args.classification:
        log("ERROR: SMOTE can only be used with --classification mode.")
        exit(1)

    main(args.config, classification=args.classification, smote=args.smote, name=args.name)