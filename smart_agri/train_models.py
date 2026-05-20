"""
train_models.py — End-to-end training pipeline.

Run once before launching the GUI:
    python train_models.py

Outputs
  models/   — serialized joblib artifacts (.pkl)
  results/  — evaluation plots (.png) + metrics_summary.json
  data/     — generated crop_data.csv
"""

import os
import sys
import json

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.preprocessing import load_and_preprocess
from src.models import (
    train_decision_tree,
    train_kmeans_clustering,
    train_linear_regression,
    save_models,
)

DATA_PATH   = os.path.join(_ROOT, 'data', 'crop_data.csv')
MODELS_DIR  = os.path.join(_ROOT, 'models')
RESULTS_DIR = os.path.join(_ROOT, 'results')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(_ROOT, 'data'), exist_ok=True)


def main():
    sep = '=' * 62
    print(sep)
    print('  Smart Agriculture DSS — Model Training Pipeline')
    print(sep)

    # ── 1. Data ─────────────────────────────────────────────
    print('\n[1/4]  Loading & preprocessing data …')
    data = load_and_preprocess(DATA_PATH)
    df   = data['df']
    print(f'       Dataset  : {df.shape[0]} samples × {len(data["feature_names"])} features')
    print(f'       Classes  : {len(data["label_encoder"].classes_)} crops')
    print(f'       Split    : {len(data["X_train"])} train / {len(data["X_test"])} test')

    # ── 2. Decision Tree ─────────────────────────────────────
    print('\n[2/4]  Training Decision Tree Classifier …')
    dt, dt_m, _, _ = train_decision_tree(
        data['X_train'], data['y_cls_train'],
        data['X_test'],  data['y_cls_test'],
        data['feature_names'], data['label_encoder'],
        RESULTS_DIR,
    )
    print(f'       Accuracy  : {dt_m["accuracy"]:.4f}')
    print(f'       Precision : {dt_m["precision"]:.4f}')
    print(f'       Recall    : {dt_m["recall"]:.4f}')
    print(f'       F1-Score  : {dt_m["f1"]:.4f}')

    # ── 3. KMeans Clustering ─────────────────────────────────
    print('\n[3/4]  Training KMeans Clustering Model …')
    km, _, km_m, _ = train_kmeans_clustering(
        data['X_all_scaled'], data['X_raw'], RESULTS_DIR, n_clusters=4,
    )
    print(f'       Clusters          : {km_m["n_clusters"]}')
    print(f'       Silhouette Score  : {km_m["silhouette_score"]:.4f}')
    print(f'       Best k (sil.)     : {km_m["best_k_by_silhouette"]}')

    # ── 4. Linear Regression ─────────────────────────────────
    # Uses X_train_reg = [7 scaled features | normalized crop label]
    # Crop label is the ground-truth at training time (DT prediction at inference).
    print('\n[4/4]  Training Linear Regression Model …')
    reg, _, reg_m, _, std_res = train_linear_regression(
        data['X_train_reg'], data['y_reg_train'],
        data['X_test_reg'],  data['y_reg_test'],
        RESULTS_DIR,
    )
    print(f'       RMSE             : {reg_m["rmse"]:,.2f} kg/ha')
    print(f'       MAE              : {reg_m["mae"]:,.2f} kg/ha')
    print(f'       R2               : {reg_m["r2"]:.4f}')
    print(f'       Residual Std Dev : {std_res:,.2f} kg/ha')

    # ── Serialize ────────────────────────────────────────────
    print('\n  Saving serialized model artifacts …')
    save_models({
        'decision_tree':     dt,
        'kmeans':            km,
        'linear_regression': reg,
        'scaler':            data['scaler'],
        'label_encoder':     data['label_encoder'],
        'crop_ohe':          data['crop_ohe'],
        'reg_std_residual':  std_res,
    }, MODELS_DIR)

    # ── Metrics summary ──────────────────────────────────────
    summary = {
        'decision_tree':    dt_m,
        'kmeans':           km_m,
        'linear_regression': reg_m,
    }
    out_path = os.path.join(RESULTS_DIR, 'metrics_summary.json')
    with open(out_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'  Metrics summary -> {out_path}')

    print(f'\n{sep}')
    print('  Training complete!')
    print(f'  Models  -> {MODELS_DIR}')
    print(f'  Results -> {RESULTS_DIR}')
    print('  Launch GUI: python -m src.gui')
    print(sep)


if __name__ == '__main__':
    main()
