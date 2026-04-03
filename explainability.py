"""
Explainability Module for Parkinson's Disease Detection.

Provides interpretable insights into ANFIS model decisions:
  • Fuzzy rule extraction in human-readable IF-THEN format
  • Permutation-based feature importance
  • Gaussian membership function visualisation
  • 2-D decision surface plots for top feature pairs
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.inspection import permutation_importance as sklearn_perm_imp

from anfis_model import ANFIS


# ─── Fuzzy Rule Extraction ──────────────────────────────────────────────────


def extract_fuzzy_rules(
    anfis_model: ANFIS,
    feature_names: list = None,
    top_k: int = 10,
) -> list:
    """
    Convert ANFIS rules into human-readable IF-THEN strings.

    Delegates to ANFIS.get_rules() and enriches with context.

    Returns:
        List of rule strings.
    """
    return anfis_model.get_rules(feature_names=feature_names, top_k=top_k)


# ─── Feature Importance (Permutation) ───────────────────────────────────────


def compute_feature_importance(
    anfis_model: ANFIS,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    sel_idx: np.ndarray = None,
    scaler=None,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Permutation feature importance for the ANFIS model.

    Args:
        anfis_model: Trained ANFIS model.
        X: Full feature matrix (n_samples, n_features).
        y: True labels.
        feature_names: Feature names.
        sel_idx: Indices of selected features used by ANFIS.
        scaler: Fitted scaler for the selected features.
        n_repeats: Number of permutation repeats.

    Returns:
        DataFrame with columns (Feature, Importance, Std) sorted descending.
    """
    if sel_idx is not None:
        X_sel = X[:, sel_idx]
        names = [feature_names[i] for i in sel_idx]
    else:
        X_sel = X
        names = feature_names

    if scaler is not None:
        X_scaled = scaler.transform(X_sel)
    else:
        X_scaled = X_sel

    # Wrap ANFIS as a sklearn-compatible scorer
    class _ANFISWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, X_arr):
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_arr, dtype=torch.float32)
                prob = self.model(X_t).numpy()
            return (prob >= 0.5).astype(int)

        def score(self, X_arr, y_true):
            pred = self.predict(X_arr)
            return float(np.mean(pred == y_true))

    wrapper = _ANFISWrapper(anfis_model)

    result = sklearn_perm_imp(
        wrapper, X_scaled, y,
        n_repeats=n_repeats,
        scoring="accuracy",
        random_state=42,
    )

    df = pd.DataFrame({
        "Feature": names,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df


# ─── Membership Function Plots ──────────────────────────────────────────────


def plot_membership_functions(
    anfis_model: ANFIS,
    feature_names: list = None,
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot Gaussian membership functions for each input feature.

    Returns:
        matplotlib Figure with one subplot per feature.
    """
    n_inputs = anfis_model.n_inputs
    n_mfs = anfis_model.n_mfs
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_inputs)]

    labels = (
        ["LOW", "HIGH"]
        if n_mfs == 2
        else [f"MF{j}" for j in range(n_mfs)]
    )
    colors = plt.cm.Set1(np.linspace(0, 1, n_mfs))

    cols = 4
    rows = int(np.ceil(n_inputs / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False
    )

    with torch.no_grad():
        centers = anfis_model.mf.center.numpy()  # (n_inputs, n_mfs)
        sigmas = anfis_model.mf.sigma.numpy()

    for i in range(n_inputs):
        ax = axes[i // cols][i % cols]
        lo = centers[i].min() - 3 * sigmas[i].max()
        hi = centers[i].max() + 3 * sigmas[i].max()
        x = np.linspace(lo, hi, n_points)

        for j in range(n_mfs):
            mu = np.exp(-0.5 * ((x - centers[i, j]) / sigmas[i, j]) ** 2)
            ax.plot(x, mu, color=colors[j], linewidth=2, label=labels[j])
            ax.axvline(
                centers[i, j], color=colors[j], ls="--", alpha=0.4
            )

        ax.set_title(feature_names[i], fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("μ")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for idx in range(n_inputs, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(
        "ANFIS Gaussian Membership Functions",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return fig


# ─── Decision Surface ───────────────────────────────────────────────────────


def plot_decision_surface(
    anfis_model: ANFIS,
    X: np.ndarray,
    y: np.ndarray,
    feature_pair: tuple = (0, 1),
    feature_names: list = None,
    sel_idx: np.ndarray = None,
    scaler=None,
    resolution: int = 100,
) -> plt.Figure:
    """
    Plot 2-D decision surface for a pair of features, with other
    features held at their mean values.

    Args:
        feature_pair: Tuple of two column indices (relative to selected
                      features, not full feature set).
    """
    if sel_idx is not None:
        X_sel = X[:, sel_idx]
    else:
        X_sel = X

    if scaler is not None:
        X_scaled = scaler.transform(X_sel)
    else:
        X_scaled = X_sel

    fi, fj = feature_pair
    if feature_names is None:
        feature_names = [f"x{k}" for k in range(X_scaled.shape[1])]

    x_min, x_max = X_scaled[:, fi].min() - 0.5, X_scaled[:, fi].max() + 0.5
    y_min, y_max = X_scaled[:, fj].min() - 0.5, X_scaled[:, fj].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Build grid input with means for other features
    mean_vals = X_scaled.mean(axis=0)
    grid = np.tile(mean_vals, (xx.ravel().shape[0], 1))
    grid[:, fi] = xx.ravel()
    grid[:, fj] = yy.ravel()

    anfis_model.eval()
    with torch.no_grad():
        grid_t = torch.tensor(grid, dtype=torch.float32)
        Z = anfis_model(grid_t).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn_r", alpha=0.8)
    plt.colorbar(cf, ax=ax, label="PD Risk Score")

    # Scatter actual points
    for label, color, marker, name in [
        (0, "#10b981", "o", "Healthy"),
        (1, "#ef4444", "s", "PD"),
    ]:
        mask = y == label
        ax.scatter(
            X_scaled[mask, fi],
            X_scaled[mask, fj],
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.5,
            s=40,
            label=name,
            alpha=0.8,
        )

    ax.set_xlabel(feature_names[fi], fontsize=11)
    ax.set_ylabel(feature_names[fj], fontsize=11)
    ax.set_title(
        f"ANFIS Decision Surface — {feature_names[fi]} vs {feature_names[fj]}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best")
    plt.tight_layout()
    return fig
