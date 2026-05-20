"""
models.py — Train, evaluate, and serialize all three AI models.
  1. Decision Tree Classifier  — crop recommendation
  2. KMeans Clustering         — soil zone segmentation
  3. Linear Regression         — yield prediction
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from src.utils import CLUSTER_PROFILES

# ─────────────────────────────────────────────────────────────
# Decision Tree
# ─────────────────────────────────────────────────────────────

def train_decision_tree(X_train, y_train, X_test, y_test, feature_names, label_encoder, results_dir):
    clf = DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred   = clf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # Feature importance bar chart
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(feature_names)), importances[idx], color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Importance Score', fontsize=11)
    ax.set_title('Decision Tree — Feature Importance for Crop Recommendation', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(0.98, 0.98, f'Accuracy: {accuracy:.3f}  |  F1: {f1:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    path = os.path.join(results_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return clf, metrics, importances, path


# ─────────────────────────────────────────────────────────────
# KMeans Clustering
# ─────────────────────────────────────────────────────────────

def train_kmeans_clustering(X_all_scaled, X_raw, results_dir, n_clusters=4):
    # Silhouette analysis to find best k
    k_range = range(2, 9)
    sil_scores = []
    for k in k_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_all_scaled)
        sil_scores.append(silhouette_score(X_all_scaled, labels_tmp))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_all_scaled)
    sil = silhouette_score(X_all_scaled, cluster_labels)

    # PCA for 2-D scatter
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X_all_scaled)

    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
               '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(n_clusters):
        mask = cluster_labels == i
        lbl  = CLUSTER_PROFILES.get(i, {}).get('name', f'Zone {i}')
        axes[0].scatter(X2d[mask, 0], X2d[mask, 1],
                        c=palette[i % len(palette)], label=f'Zone {i}: {lbl}',
                        alpha=0.55, s=18, edgecolors='none')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)', fontsize=10)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)', fontsize=10)
    axes[0].set_title('KMeans — Soil Zone Cluster Distribution (PCA)', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(alpha=0.25)
    axes[0].text(0.02, 0.02, f'Silhouette Score: {sil:.3f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    axes[1].plot(list(k_range), sil_scores, 'bo-', linewidth=2, markersize=7)
    axes[1].axvline(x=n_clusters, color='red', linestyle='--', linewidth=1.5,
                    label=f'Selected k = {n_clusters}')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=10)
    axes[1].set_ylabel('Silhouette Score', fontsize=10)
    axes[1].set_title('Silhouette Score vs. Number of Clusters', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    plt.suptitle('KMeans Clustering — Soil Profile Segmentation', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(results_dir, 'cluster_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': sil,
        'best_k_by_silhouette': int(k_range[int(np.argmax(sil_scores))]),
        'all_silhouette_scores': {k: round(s, 4) for k, s in zip(k_range, sil_scores)},
    }
    return kmeans, cluster_labels, metrics, path


# ─────────────────────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train, X_test, y_test, results_dir):
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred      = reg.predict(X_test)
    residuals   = y_test - y_pred
    std_residual = float(np.std(residuals))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'std_residual': std_residual}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.45, s=15, color='steelblue', edgecolors='none')
    axes[0].axhline(0, color='red', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Fitted Values (kg/ha)', fontsize=10)
    axes[0].set_ylabel('Residuals', fontsize=10)
    axes[0].set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
    axes[0].grid(alpha=0.25)

    # Actual vs Predicted
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    axes[1].scatter(y_test, y_pred, alpha=0.45, s=15, color='seagreen', edgecolors='none')
    axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect fit')
    axes[1].set_xlabel('Actual Yield (kg/ha)', fontsize=10)
    axes[1].set_ylabel('Predicted Yield (kg/ha)', fontsize=10)
    axes[1].set_title(f'Actual vs Predicted  (R² = {r2:.3f})', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    # Residual histogram
    axes[2].hist(residuals, bins=35, color='coral', edgecolor='white', alpha=0.85)
    axes[2].axvline(0, color='black', linewidth=1.5, linestyle='--')
    axes[2].set_xlabel('Residuals', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].set_title('Residual Distribution', fontsize=11, fontweight='bold')
    axes[2].grid(alpha=0.25)

    plt.suptitle(
        f'Linear Regression — Crop Yield Prediction'
        f'  |  RMSE={rmse:,.1f}  MAE={mae:,.1f}  R²={r2:.3f}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    path = os.path.join(results_dir, 'residual_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return reg, y_pred, metrics, path, std_residual


# ─────────────────────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────────────────────

def save_models(models_dict, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for name, obj in models_dict.items():
        p = os.path.join(models_dir, f'{name}.pkl')
        joblib.dump(obj, p)
        print(f'  Saved -> {p}')


def load_models(models_dir):
    names = [
        'decision_tree', 'kmeans', 'linear_regression',
        'scaler', 'label_encoder', 'crop_ohe', 'reg_std_residual',
    ]
    loaded = {}
    for name in names:
        p = os.path.join(models_dir, f'{name}.pkl')
        if os.path.exists(p):
            loaded[name] = joblib.load(p)
    return loaded
