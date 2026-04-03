"""
Training & Model Comparison Pipeline.

Downloads the UCI Parkinson's dataset, preprocesses features, and runs
10-fold stratified cross-validation on four models:

  1. ANFIS  (custom PyTorch — this project's main model)
  2. SVM    (RBF kernel, GridSearchCV)
  3. Random Forest  (100 trees)
  4. Neural Network  (3-layer MLP, PyTorch)

Outputs a comparison table and saves the best model checkpoints.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils.helpers import (
    load_uci_dataset,
    get_feature_matrix,
    MODELS_DIR,
    plot_comparison_table,
    plot_feature_importance,
)
from anfis_model import ANFIS, select_top_features, pso_optimize_mf

warnings.filterwarnings("ignore")


# ─── Neural Network Baseline ────────────────────────────────────────────────


class ParkinsonsNN(nn.Module):
    """3-layer MLP for Parkinson's classification."""

    def __init__(self, n_inputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Individual Trainers ────────────────────────────────────────────────────


def _oversample_minority(X, y, ratio=0.8):
    """
    Random oversampling of minority class.
    ratio=1.0 → full balance, ratio=0.8 → minority reaches 80% of majority.
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    target = int(max_count * ratio)
    X_parts, y_parts = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < target:
            deficit = target - cnt
            idx = np.where(y == cls)[0]
            dup_idx = np.random.choice(idx, size=deficit, replace=True)
            X_parts.append(X[dup_idx])
            y_parts.append(y[dup_idx])
    return np.vstack(X_parts), np.concatenate(y_parts)


def train_anfis(
    X_train, y_train,
    feature_names: list,
    n_features: int = 12,       # more signal retained
    n_mfs: int = 3,             # finer fuzzy granularity
    epochs: int = 300,          # more convergence time
    use_pso: bool = True,
):
    """
    Select features → init from data → PSO → hybrid train → prune → fine-tune.

    Returns:
        (model, selected_indices, scaler, selected_names)
    """
    # 1. Feature selection
    X_sel, sel_names, sel_idx = select_top_features(
        X_train, y_train, feature_names, k=n_features,
    )

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # 3. Build model
    model = ANFIS(n_inputs=n_features, n_mfs=n_mfs)

    # 4. Init MF centers/widths from data percentiles
    model.init_from_data(X_scaled)

    # 5. PSO initialisation with stronger search
    if use_pso:
        model = pso_optimize_mf(
            model, X_scaled, y_train,
            n_particles=25,
            iters=40,
            verbose=False,
        )

    # 6. Main hybrid training
    model.hybrid_train(
        X_scaled, y_train,
        epochs=epochs,
        lr=0.005,
        verbose=False,
    )

    # 7. Less aggressive pruning
    model.prune_rules(X_scaled, threshold=0.002, keep_min=32)

    # 8. Fine-tune after pruning with lower LR
    model.hybrid_train(
        X_scaled, y_train,
        epochs=100,
        lr=0.001,
        verbose=False,
    )

    return model, sel_idx, scaler, sel_names


def train_svm(X_train, y_train):
    """Train SVM with RBF kernel and GridSearch."""
    param_grid = {
        "C": [1, 10],
        "gamma": ["scale", 0.01],
    }
    svm = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    svm.fit(X_train, y_train)
    return svm.best_estimator_


def train_rf(X_train, y_train):
    """Train Random Forest with tuned hyperparameters."""
    param_grid = {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=100, random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.best_estimator_


def train_nn(X_train, y_train, epochs: int = 100, lr: float = 0.001):
    """Train 3-layer MLP."""
    n_inputs = X_train.shape[1]
    model = ParkinsonsNN(n_inputs)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        pred = model(X_t)
        loss = criterion(pred, y_t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model


# ─── Evaluation Helpers ─────────────────────────────────────────────────────


def _evaluate(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "F1-Score": f1,
        "AUC-ROC": auc,
    }


def _predict_anfis(model, X, sel_idx, scaler):
    """Predict with ANFIS (feature selection + scaling applied)."""
    X_sel = X[:, sel_idx]
    X_scaled = scaler.transform(X_sel)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prob = model(X_t).numpy()
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def _predict_nn(model, X):
    """Predict with PyTorch NN."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        prob = model(X_t).numpy()
    pred = (prob >= 0.5).astype(int)
    return pred, prob


# ─── Cross-Validation Pipeline ──────────────────────────────────────────────


def cross_validate_all(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_folds: int = 10,
) -> pd.DataFrame:
    """
    Run 10-fold Stratified CV for ANFIS, SVM, RF, NN.

    Returns:
        DataFrame of averaged metrics per model.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    model_names = ["ANFIS", "SVM (RBF)", "Random Forest", "Neural Network"]
    all_metrics = {m: [] for m in model_names}
    all_times = {m: [] for m in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"  Fold {fold}/{n_folds}")
        print(f"{'='*50}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale full features for SVM / RF / NN
        scaler_full = StandardScaler()
        X_train_sc = scaler_full.fit_transform(X_train)
        X_test_sc = scaler_full.transform(X_test)

        # ── ANFIS ────────────────────────────────────────────────────
        t0 = time.time()
        anfis_model, sel_idx, anfis_scaler, sel_names = train_anfis(
            X_train, y_train, feature_names,
            n_features=12, n_mfs=3, epochs=300, use_pso=True,
        )
        t_anfis = time.time() - t0
        y_pred_a, y_prob_a = _predict_anfis(
            anfis_model, X_test, sel_idx, anfis_scaler,
        )
        all_metrics["ANFIS"].append(_evaluate(y_test, y_pred_a, y_prob_a))
        all_times["ANFIS"].append(t_anfis)
        print(f"  ANFIS         done  ({t_anfis:.1f}s)")

        # ── SVM ──────────────────────────────────────────────────────
        t0 = time.time()
        svm_model = train_svm(X_train_sc, y_train)
        t_svm = time.time() - t0
        y_pred_s = svm_model.predict(X_test_sc)
        y_prob_s = svm_model.predict_proba(X_test_sc)[:, 1]
        all_metrics["SVM (RBF)"].append(_evaluate(y_test, y_pred_s, y_prob_s))
        all_times["SVM (RBF)"].append(t_svm)
        print(f"  SVM (RBF)     done  ({t_svm:.1f}s)")

        # ── Random Forest ────────────────────────────────────────────
        t0 = time.time()
        rf_model = train_rf(X_train_sc, y_train)
        t_rf = time.time() - t0
        y_pred_r = rf_model.predict(X_test_sc)
        y_prob_r = rf_model.predict_proba(X_test_sc)[:, 1]
        all_metrics["Random Forest"].append(
            _evaluate(y_test, y_pred_r, y_prob_r)
        )
        all_times["Random Forest"].append(t_rf)
        print(f"  Random Forest done  ({t_rf:.1f}s)")

        # ── Neural Network ───────────────────────────────────────────
        t0 = time.time()
        nn_model = train_nn(X_train_sc, y_train, epochs=200)
        t_nn = time.time() - t0
        y_pred_n, y_prob_n = _predict_nn(nn_model, X_test_sc)
        all_metrics["Neural Network"].append(
            _evaluate(y_test, y_pred_n, y_prob_n)
        )
        all_times["Neural Network"].append(t_nn)
        print(f"  Neural Net    done  ({t_nn:.1f}s)")

    # Average across folds
    rows = []
    for m in model_names:
        avg = {}
        for key in all_metrics[m][0]:
            avg[key] = np.mean([d[key] for d in all_metrics[m]])
        avg["Avg Train Time (s)"] = np.mean(all_times[m])
        rows.append(avg)

    df = pd.DataFrame(rows, index=model_names)
    return df


# ─── Full Pipeline ──────────────────────────────────────────────────────────


def run_full_pipeline():
    """
    End-to-end pipeline:
      1. Download / load UCI dataset
      2. 10-fold CV on all models
      3. Train final models on full data
      4. Save models + scaler
      5. Print results
    """
    print("=" * 60)
    print("  Parkinson's Disease Detection — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_uci_dataset()
    X, y, feature_names = get_feature_matrix(df)
    print(f"\n  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"  PD: {int(y.sum())}  |  Healthy: {int(len(y) - y.sum())}\n")

    # 2. Cross-validate
    results_df = cross_validate_all(X, y, feature_names, n_folds=5)

    print("\n" + "=" * 60)
    print("  5-Fold Cross-Validation Results")
    print("=" * 60)
    print(results_df.round(4).to_string())

    # 3. Train final models on full data
    print("\n\n  Training final models on full dataset...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)

    # ANFIS
    anfis_model, sel_idx, anfis_scaler, sel_names = train_anfis(
        X, y, feature_names, n_features=12, n_mfs=3, epochs=350, use_pso=True,
    )
    torch.save(
        {
            "model_state": anfis_model.state_dict(),
            "n_inputs": anfis_model.n_inputs,
            "n_mfs": anfis_model.n_mfs,
            "n_rules": anfis_model.n_rules,
            "sel_idx": sel_idx,
            "sel_names": sel_names,
            "scaler_mean": anfis_scaler.mean_,
            "scaler_scale": anfis_scaler.scale_,
        },
        os.path.join(MODELS_DIR, "anfis_model.pt"),
    )

    # SVM
    svm_model = train_svm(X_scaled, y)
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))

    # RF
    rf_model = train_rf(X_scaled, y)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))

    # NN
    nn_model = train_nn(X_scaled, y, epochs=150)
    torch.save(nn_model.state_dict(), os.path.join(MODELS_DIR, "nn_model.pt"))

    # Full scaler
    joblib.dump(scaler_full, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Feature names
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    print(f"\n  Models saved to {MODELS_DIR}/")

    # 4. Plots
    results_dict = {}
    for model_name in results_df.index:
        results_dict[model_name] = results_df.loc[model_name].to_dict()

    fig_comp = plot_comparison_table(results_dict)
    fig_comp.savefig(
        os.path.join(MODELS_DIR, "comparison_chart.png"), dpi=150
    )
    print("  Comparison chart saved.\n")

    # RF feature importance
    fig_imp = plot_feature_importance(
        rf_model.feature_importances_, feature_names,
        title="Random Forest — Feature Importance",
    )
    fig_imp.savefig(
        os.path.join(MODELS_DIR, "rf_feature_importance.png"), dpi=150
    )

    return results_df


# ─── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_full_pipeline()
