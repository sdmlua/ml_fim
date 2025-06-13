import os
import toml
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path

def log(msg):
    print(f"[MLFIM] {msg}")

def load_raster_as_array(path):
    log(f"Reading raster: {path}")
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
    return array, nodata

def create_dataframe(feature_paths):
    log("Creating DataFrame from rasters…")
    arrays = []
    names = []
    nodata_map = {}

    for name, path in feature_paths.items():
        arr, nodata = load_raster_as_array(path)
        if nodata is not None:
            log(f"Replacing nodata for '{name}' ({nodata}) with NaN")
            arr[arr == nodata] = np.nan
        arrays.append(arr)
        names.append(name)

    stacked = np.stack(arrays, axis=-1)
    flat = stacked.reshape(-1, stacked.shape[-1])
    df = pd.DataFrame(flat, columns=names)
    return df

def get_next_output_id(base_path):
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("data_")]
    max_id = 0
    for name in existing:
        try:
            num = int(name.replace("data_", ""))
            if num > max_id:
                max_id = num
        except ValueError:
            continue
    return max_id + 1

def main(toml_path: str):
    log("Loading configuration…")
    cfg = toml.load(toml_path)
    root = Path(".").resolve()
    huc8 = cfg["huc8"]

    feature_paths = {
        k: root / "data/raw" / huc8 / ("output" if k == "flood_depth" else "input") / Path(v).name
        for k, v in cfg["features"].items()
    }

    df = create_dataframe(feature_paths)

    out_base = Path("data/processed") / huc8
    out_dir = out_base / f"data_{get_next_output_id(out_base)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Saving → {out_dir}")
    df.to_pickle(out_dir / "full_data.pkl")

    info_path = out_dir / "info.txt"
    log(f"Writing statistics summary → {info_path}")

    with open(info_path, "w") as f:
        f.write("MLFIM Data Summary\n" + "=" * 40 + "\n\n")
        f.write(f"HUC8: {huc8}\n")
        f.write("Features:\n" + "-" * 25 + "\n")
        for k, v in feature_paths.items():
            f.write(f"{k:15s}: {v}\n")
        f.write("\nData Statistics:\n" + "-" * 30 + "\n")
        f.write(df.describe().to_string())
        f.write("\n")

    log("✅ Preprocessing complete.")

if __name__ == "__main__":
    main("config/inputs.toml")
