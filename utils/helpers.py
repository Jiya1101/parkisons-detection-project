"""
Shared utilities for the Parkinson's Disease Detection System.

Includes dataset download, plotting helpers, feature mappings, and audio I/O.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for compatibility

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
)
UCI_DATASET_PATH = os.path.join(DATA_DIR, "parkinsons.data")


# ─── Feature Name Mappings ───────────────────────────────────────────────────

# The 22 voice features from the UCI Parkinson's dataset
UCI_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",       # Average vocal fundamental frequency
    "MDVP:Fhi(Hz)",      # Maximum vocal fundamental frequency
    "MDVP:Flo(Hz)",      # Minimum vocal fundamental frequency
    "MDVP:Jitter(%)",    # Jitter percentage
    "MDVP:Jitter(Abs)",  # Absolute jitter
    "MDVP:RAP",          # Relative amplitude perturbation
    "MDVP:PPQ",          # Five-point period perturbation quotient
    "Jitter:DDP",        # Average absolute difference of differences of consecutive periods
    "MDVP:Shimmer",      # Shimmer (local)
    "MDVP:Shimmer(dB)",  # Shimmer in dB
    "Shimmer:APQ3",      # Three-point amplitude perturbation quotient
    "Shimmer:APQ5",      # Five-point amplitude perturbation quotient
    "MDVP:APQ",          # Eleven-point amplitude perturbation quotient (APQ11)
    "Shimmer:DDA",       # Average absolute difference of differences of consecutive amplitudes
    "NHR",               # Noise-to-harmonics ratio
    "HNR",               # Harmonics-to-noise ratio
    "RPDE",              # Recurrence period density entropy
    "DFA",               # Detrended fluctuation analysis
    "spread1",           # Nonlinear measure of fundamental frequency variation
    "spread2",           # Nonlinear measure of fundamental frequency variation
    "D2",                # Correlation dimension
    "PPE",               # Pitch period entropy
]

# Short feature labels for plots
FEATURE_SHORT_NAMES = [
    "F0_mean", "F0_max", "F0_min",
    "Jitter%", "Jitter_abs", "RAP", "PPQ", "DDP",
    "Shimmer", "Shimmer_dB", "APQ3", "APQ5", "APQ11", "DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# Normal ranges (approximate, for display purposes)
FEATURE_NORMAL_RANGES = {
    "MDVP:Fo(Hz)":      (100, 260),
    "MDVP:Fhi(Hz)":     (100, 600),
    "MDVP:Flo(Hz)":     (60, 240),
    "MDVP:Jitter(%)":   (0.0, 1.04),
    "MDVP:Jitter(Abs)": (0.0, 0.00008),
    "MDVP:RAP":         (0.0, 0.006),
    "MDVP:PPQ":         (0.0, 0.006),
    "Jitter:DDP":       (0.0, 0.018),
    "MDVP:Shimmer":     (0.0, 0.035),
    "MDVP:Shimmer(dB)": (0.0, 0.350),
    "Shimmer:APQ3":     (0.0, 0.018),
    "Shimmer:APQ5":     (0.0, 0.022),
    "MDVP:APQ":         (0.0, 0.030),
    "Shimmer:DDA":      (0.0, 0.054),
    "NHR":              (0.0, 0.020),
    "HNR":              (20.0, 35.0),
    "RPDE":             (0.25, 0.60),
    "DFA":              (0.55, 0.75),
    "spread1":          (-7.0, -5.0),
    "spread2":          (0.0, 0.3),
    "D2":               (1.5, 3.0),
    "PPE":              (0.0, 0.2),
}


# ─── Dataset Utilities ──────────────────────────────────────────────────────

def download_uci_dataset(force: bool = False) -> str:
    """
    Download the UCI Parkinson's dataset if not already present.
    
    Args:
        force: If True, re-download even if file exists.
        
    Returns:
        Path to the downloaded dataset file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(UCI_DATASET_PATH) and not force:
        print(f"[INFO] Dataset already exists at {UCI_DATASET_PATH}")
        return UCI_DATASET_PATH
    
    print(f"[INFO] Downloading UCI Parkinson's dataset...")
    try:
        urllib.request.urlretrieve(UCI_DATASET_URL, UCI_DATASET_PATH)
        print(f"[INFO] Dataset saved to {UCI_DATASET_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        raise
    
    return UCI_DATASET_PATH


def load_uci_dataset() -> pd.DataFrame:
    """
    Load the UCI Parkinson's dataset as a pandas DataFrame.
    Downloads it first if not present.
    
    Returns:
        DataFrame with all features and 'status' column (1=PD, 0=healthy).
    """
    path = download_uci_dataset()
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"[INFO] Class distribution: PD={df['status'].sum()}, Healthy={len(df) - df['status'].sum()}")
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extract feature matrix X and labels y from the UCI dataset.
    
    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, 22)
        y (np.ndarray): Label vector (1=PD, 0=healthy)
        feature_names (list): List of feature column names
    """
    # Drop 'name' and 'status' columns
    feature_cols = [c for c in df.columns if c not in ('name', 'status')]
    X = df[feature_cols].values.astype(np.float64)
    y = df['status'].values.astype(np.float64)
    return X, y, feature_cols


# ─── Plotting Helpers ────────────────────────────────────────────────────────

def plot_feature_importance(importances: np.ndarray, feature_names: list, 
                            title: str = "Feature Importance", top_n: int = 15):
    """
    Create a horizontal bar chart of feature importances.
    
    Args:
        importances: Array of importance scores.
        feature_names: List of feature names.
        title: Plot title.
        top_n: Number of top features to show.
        
    Returns:
        matplotlib Figure.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    
    ax.barh(range(top_n), importances[indices][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison_table(results: dict):
    """
    Create a comparison bar chart for model metrics.
    
    Args:
        results: Dict of {model_name: {metric: value}}.
        
    Returns:
        matplotlib Figure.
    """
    models = list(results.keys())
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.18
    
    colors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444']
    
    for i, model in enumerate(models):
        values = [results[model].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)], 
               edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Parkinson\'s Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_gauge_chart(score: float, title: str = "Parkinson's Risk Score"):
    """
    Create a gauge/speedometer chart for the risk score using plotly.
    
    Args:
        score: Risk score between 0 and 1.
        title: Chart title.
        
    Returns:
        plotly Figure.
    """
    import plotly.graph_objects as go
    
    # Determine color zone
    if score < 0.3:
        color = "#10b981"  # Green - low risk
        label = "Low Risk"
    elif score < 0.6:
        color = "#f59e0b"  # Amber - moderate risk
        label = "Moderate Risk"
    elif score < 0.8:
        color = "#f97316"  # Orange - elevated risk
        label = "Elevated Risk"
    else:
        color = "#ef4444"  # Red - high risk
        label = "High Risk"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': color}},
        title={'text': f"{title}<br><span style='font-size:16px;color:{color}'>{label}</span>",
               'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#334155"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.15)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': score * 100,
            },
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
    )
    
    return fig


# ─── Audio I/O ───────────────────────────────────────────────────────────────

def load_audio(file_path: str, sr: int = 22050):
    """
    Load an audio file and return the signal and sample rate.
    
    Args:
        file_path: Path to the audio file.
        sr: Target sample rate.
        
    Returns:
        Tuple of (audio_signal, sample_rate).
    """
    import librosa
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate


def save_audio(audio: np.ndarray, file_path: str, sr: int = 22050):
    """Save audio signal to a WAV file."""
    import soundfile as sf
    sf.write(file_path, audio, sr)
    print(f"[INFO] Audio saved to {file_path}")
