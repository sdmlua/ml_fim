# ml_fim

A surrogate machine learning model to improve flood depth estimation using OWP-HAND FIM depth products and other hydrological attributes.


## 🧪 Data Preprocessing

The `process_data.py` script prepares the input raster data for model training and evaluation by performing the following key steps:

### ✅ What It Does:
- Reads and flattens raster data for each HUC8 watershed and return period.
- Cleans and normalizes rasters, handling `NoData` values.
- Computes derived features like `aspect_sin` and `aspect_cos`.
- Ensures consistent shape alignment across all rasters.
- **Balances the dataset** by limiting zero flood depth values to match the count of non-zero values (1:1 ratio).
- Performs a stratified train-test split (default: 70% train, 30% test).
- Saves:
  - One combined `train.pkl` file for all training data.
  - Separate `*_test.pkl` files for each HUC8 and return period.

### ⚙️ Example Usage:
```bash
python process_data.py --input ../data/ --output ../data_processed/
```

- `--input`: Path to the base directory containing raster data folders.

- `--output`: Output directory where the processed `.pkl` files will be saved.

- `--test_size`: (Optional) Proportion of test data. Default is `0.3`.

### Output Structure
data_processed/
│
├── train.pkl                    # Combined, balanced training dataset
│
├── <HUC8>/
│   ├── 10year_test.pkl
│   ├── 50year_test.pkl
│   └── ...

### 💡 Note

- Balancing is applied **before** splitting to reduce data volume and speed up training.
- Only raster pairs with **valid and aligned shapes** are included.
