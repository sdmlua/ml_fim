import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    f1_score,
    auc
)

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Flood classification precision-recall analysis.")
parser.add_argument('--root_dir', type=str, required=True, help="Root directory with *_test.pkl files")
parser.add_argument('--output_dir', type=str, default="flood_eval_output", help="Directory to save all plots")
parser.add_argument('--log_target', action='store_true', help="Unlog predictions and targets if trained on log1p")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Load and concatenate test data
# -----------------------------
test_files = glob(os.path.join(args.root_dir, "**", "*_test.pkl"), recursive=True)
print(f"📂 Found {len(test_files)} test files.")

all_dfs = []
for path in test_files:
    try:
        df = joblib.load(path)
        if not {"flood_depth", "predicted"}.issubset(df.columns):
            print(f"⚠️ Skipping {path}: missing flood_depth or predicted")
            continue

        if args.log_target:
            df["flood_depth"] = np.expm1(df["flood_depth"])
            df["predicted"] = np.expm1(df["predicted"])

        all_dfs.append(df[["flood_depth", "predicted"]])
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")

if not all_dfs:
    print("❌ No valid test data found.")
    exit()

df_all = pd.concat(all_dfs, ignore_index=True).dropna()
print(f"✅ Combined rows: {len(df_all)}")

# -----------------------------
# Binary classification setup
# -----------------------------
y_true_bin = (df_all["flood_depth"] > 0.01).astype(int)
y_score = df_all["predicted"].values

# -----------------------------
# Compute PR curve and F1 scores
# -----------------------------
precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
pr_auc = auc(recall, precision)

# -----------------------------
# Plot Precision–Recall Curve
# -----------------------------
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.axvline(recall[best_idx], color='gray', linestyle='--', label=f'Best F1 @ {best_thresh:.4f}')
plt.axhline(precision[best_idx], color='gray', linestyle='--')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

pr_curve_path = os.path.join(args.output_dir, "precision_recall_curve.png")
plt.savefig(pr_curve_path, dpi=300)
print(f"✅ Saved PR curve to: {pr_curve_path}")
plt.close()

# -----------------------------
# Plot Precision & Recall vs Threshold (only between 0 and 1)
# -----------------------------
valid_idx = (thresholds >= 0) & (thresholds <= 1)
thresholds_limited = thresholds[valid_idx]
precision_limited = precision[:-1][valid_idx]
recall_limited = recall[:-1][valid_idx]

plt.figure(figsize=(8, 6))
plt.plot(thresholds_limited, precision_limited, label="Precision", color="blue")
plt.plot(thresholds_limited, recall_limited, label="Recall", color="green")
if 0 <= best_thresh <= 1:
    plt.axvline(best_thresh, color='gray', linestyle='--', label=f'Best F1 @ {best_thresh:.4f}')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold (0–1 m)")
plt.legend()
plt.grid(True)
plt.tight_layout()

pr_vs_thresh_path = os.path.join(args.output_dir, "precision_recall_vs_threshold.png")
plt.savefig(pr_vs_thresh_path, dpi=300)
print(f"✅ Saved Precision/Recall vs Threshold to: {pr_vs_thresh_path}")
plt.close()

# -----------------------------
# Function to plot confusion matrix
# -----------------------------
def plot_conf_matrix(y_true, y_pred, threshold, suffix):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0

    precision_pct = precision_val * 100
    recall_pct = recall_val * 100

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Flood", "Flood"])
    disp.plot(cmap="Blues", values_format='d')

    plt.title(
        f"Confusion Matrix (threshold = {threshold:.4f})\n"
        f"Precision = {precision_pct:.2f}% | Recall = {recall_pct:.2f}%",
        fontsize=11
    )
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, f"confusion_matrix_{suffix}.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved confusion matrix to: {out_path}")
    plt.close()

# -----------------------------
# Plot confusion matrix at threshold = 0.01
# -----------------------------
y_pred_fixed = (y_score > 0.01).astype(int)
plot_conf_matrix(y_true_bin, y_pred_fixed, 0.01, "threshold_0p01")

# -----------------------------
# Plot confusion matrix at best F1 threshold (if between 0–1)
# -----------------------------
if 0 <= best_thresh <= 1:
    y_pred_best = (y_score > best_thresh).astype(int)
    plot_conf_matrix(y_true_bin, y_pred_best, best_thresh, "best_threshold")
else:
    print(f"⚠️ Skipping best-threshold confusion matrix: best threshold ({best_thresh:.4f}) outside [0, 1]")
