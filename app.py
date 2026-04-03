"""
Streamlit Web UI for Parkinson's Disease Early Detection.

Sections:
  1. Header & system status
  2. Audio input — file upload (.wav) or microphone recording
  3. Feature display — extracted biomarkers with normal-range indicators
  4. Risk assessment — gauge chart (Plotly) with color zones
  5. Fuzzy rules — top fired IF-THEN rules
  6. Model comparison — metrics table + bar chart
  7. MF plots — interactive membership function visualisations
"""

import os
import io
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.helpers import (
    MODELS_DIR,
    UCI_FEATURE_NAMES,
    FEATURE_SHORT_NAMES,
    FEATURE_NORMAL_RANGES,
    create_gauge_chart,
    load_uci_dataset,
    get_feature_matrix,
)
from preprocess import load_and_preprocess, load_and_preprocess_bytes
from feature_extraction import extract_features
from anfis_model import ANFIS
from explainability import (
    extract_fuzzy_rules,
    compute_feature_importance,
    plot_membership_functions,
    plot_decision_surface,
)


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Parkinson's Disease Detection — ANFIS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark glass cards */
.glass-card {
    background: rgba(30, 30, 50, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* KPI metric row */
.kpi-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
}
.kpi-card {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-card .value {
    font-size: 1.7rem; font-weight: 700; color: #a5b4fc;
}
.kpi-card .label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #94a3b8; margin-top: 0.2rem;
}

/* Feature table colouring */
.feat-normal { color: #10b981; font-weight: 600; }
.feat-abnormal { color: #f87171; font-weight: 600; }

/* Risk tags */
.risk-low     { color: #10b981; font-weight: 700; }
.risk-mod     { color: #f59e0b; font-weight: 700; }
.risk-elev    { color: #f97316; font-weight: 700; }
.risk-high    { color: #ef4444; font-weight: 700; }

/* Rule card */
.rule-card {
    background: rgba(30, 30, 50, 0.5);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.85rem;
}

/* Section heading */
.section-head {
    font-size: 1.3rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-ready {
    background: rgba(16,185,129,0.15);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
}
.status-missing {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all saved models and scalers."""
    models = {}

    # ANFIS
    anfis_path = os.path.join(MODELS_DIR, "anfis_model.pt")
    if os.path.exists(anfis_path):
        ckpt = torch.load(anfis_path, map_location="cpu", weights_only=False)
        anfis = ANFIS(
            n_inputs=ckpt["n_inputs"],
            n_mfs=ckpt["n_mfs"],
            n_rules=ckpt["n_rules"],
        )
        anfis.load_state_dict(ckpt["model_state"])
        anfis.eval()
        models["anfis"] = anfis
        models["anfis_sel_idx"] = ckpt["sel_idx"]
        models["anfis_sel_names"] = ckpt.get("sel_names", [])

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.mean_ = ckpt["scaler_mean"]
        sc.scale_ = ckpt["scaler_scale"]
        sc.var_ = ckpt["scaler_scale"] ** 2
        sc.n_features_in_ = len(ckpt["sel_idx"])
        models["anfis_scaler"] = sc

    # Full scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        models["scaler"] = joblib.load(scaler_path)

    # SVM
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    if os.path.exists(svm_path):
        models["svm"] = joblib.load(svm_path)

    # RF
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    if os.path.exists(rf_path):
        models["rf"] = joblib.load(rf_path)

    # NN
    nn_path = os.path.join(MODELS_DIR, "nn_model.pt")
    if os.path.exists(nn_path):
        from train_compare import ParkinsonsNN
        nn = ParkinsonsNN(22)
        nn.load_state_dict(torch.load(nn_path, map_location="cpu", weights_only=True))
        nn.eval()
        models["nn"] = nn

    return models


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _risk_label(score):
    if score < 0.3:
        return "Low Risk", "risk-low"
    elif score < 0.6:
        return "Moderate Risk", "risk-mod"
    elif score < 0.8:
        return "Elevated Risk", "risk-elev"
    else:
        return "High Risk", "risk-high"


def classify_feature(name, value):
    """Return 'normal' or 'abnormal' based on known ranges."""
    if name in FEATURE_NORMAL_RANGES:
        lo, hi = FEATURE_NORMAL_RANGES[name]
        return "normal" if lo <= value <= hi else "abnormal"
    return "unknown"


# ─── Main App ───────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Navigation")
        page = st.radio(
            "Go to",
            [
                "🏠 Home — Analyse",
                "📊 Model Comparison",
                "🔍 Explainability",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            "<small style='color:#94a3b8'>"
            "Parkinson's Disease Early Detection<br>"
            "Powered by ANFIS | PyTorch<br>"
            "© 2026 Soft Computing Project"
            "</small>",
            unsafe_allow_html=True,
        )

    # ── Load models ──────────────────────────────────────────────
    models = load_models()
    has_models = "anfis" in models

    # ═══════════════════════════════════════════════════════════════
    # Page: Home — Analyse
    # ═══════════════════════════════════════════════════════════════
    if page.startswith("🏠"):
        # Header
        st.markdown(
            "<h1 style='text-align:center; "
            "background: linear-gradient(90deg, #818cf8, #c084fc, #f0abfc); "
            "-webkit-background-clip: text; "
            "-webkit-text-fill-color: transparent;'>"
            "🧠 Parkinson's Disease Early Detection"
            "</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; color:#94a3b8; margin-top:-0.5rem'>"
            "Voice-analysis-based screening powered by Adaptive Neuro-Fuzzy Inference System (ANFIS)"
            "</p>",
            unsafe_allow_html=True,
        )

        # Status
        status = "ready" if has_models else "missing"
        badge_cls = f"status-{status}"
        badge_txt = "Models loaded ✓" if has_models else "Models not trained — run train_compare.py first"
        st.markdown(
            f"<div style='text-align:center; margin-bottom:1.5rem'>"
            f"<span class='status-badge {badge_cls}'>{badge_txt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Audio input ─────────────────────────────────────────
        st.markdown("<div class='section-head'>📁 Audio Input</div>", unsafe_allow_html=True)

        col_upload, col_record = st.columns(2)

        with col_upload:
            uploaded = st.file_uploader(
                "Upload a sustained vowel 'aah' recording",
                type=["wav"],
                key="wav_upload",
            )

        with col_record:
            st.markdown("**Or record from microphone:**")
            try:
                from audio_recorder_streamlit import audio_recorder
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#6366f1",
                    neutral_color="#334155",
                    icon_size="2x",
                    pause_threshold=2.0,
                )
            except ImportError:
                audio_bytes = None
                st.info("Install `audio-recorder-streamlit` for mic recording.")

        # Determine audio source
        audio_data = None
        if uploaded is not None:
            audio_data = ("file", uploaded)
            st.audio(uploaded, format="audio/wav")
        elif audio_bytes is not None:
            audio_data = ("bytes", audio_bytes)
            st.audio(audio_bytes, format="audio/wav")

        if audio_data is None:
            st.info("Upload a WAV file or record from your microphone to begin analysis.")
            return

        if not has_models:
            st.error("⚠️ Models are not trained yet. Run `python train_compare.py` first.")
            return

        # ── Process audio ────────────────────────────────────────
        with st.spinner("🔬 Extracting features & running inference..."):
            if audio_data[0] == "file":
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_data[1].getvalue())
                    tmp_path = tmp.name
                prep = load_and_preprocess(tmp_path)
                os.unlink(tmp_path)
            else:
                prep = load_and_preprocess_bytes(audio_data[1])

            feats = extract_features(prep["audio"], sr=prep["sr"])
            feature_vector = feats["feature_vector"]
            feature_names = feats["feature_names"]

        # ── ANFIS Risk Score ─────────────────────────────────────
        anfis = models["anfis"]
        sel_idx = models["anfis_sel_idx"]
        anfis_scaler = models["anfis_scaler"]

        vec_sel = feature_vector[sel_idx]
        vec_scaled = anfis_scaler.transform(vec_sel.reshape(1, -1))
        with torch.no_grad():
            risk_score = anfis(
                torch.tensor(vec_scaled, dtype=torch.float32)
            ).item()

        # ── KPI Row ──────────────────────────────────────────────
        label, label_cls = _risk_label(risk_score)
        st.markdown(
            f"""
            <div class='kpi-row'>
                <div class='kpi-card'>
                    <div class='value'>{risk_score:.1%}</div>
                    <div class='label'>ANFIS Risk Score</div>
                </div>
                <div class='kpi-card'>
                    <div class='value {label_cls}'>{label}</div>
                    <div class='label'>Assessment</div>
                </div>
                <div class='kpi-card'>
                    <div class='value'>{prep["duration"]:.1f}s</div>
                    <div class='label'>Audio Duration</div>
                </div>
                <div class='kpi-card'>
                    <div class='value'>22</div>
                    <div class='label'>Features Extracted</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Gauge Chart ──────────────────────────────────────────
        col_gauge, col_rules = st.columns([1, 1])

        with col_gauge:
            st.markdown("<div class='section-head'>🎯 Risk Gauge</div>", unsafe_allow_html=True)
            fig_gauge = create_gauge_chart(risk_score)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_rules:
            st.markdown("<div class='section-head'>📜 Fuzzy Rules Fired</div>", unsafe_allow_html=True)
            sel_names = models.get("anfis_sel_names", [f"f{i}" for i in sel_idx])
            rules = extract_fuzzy_rules(anfis, feature_names=sel_names, top_k=8)
            for r in rules:
                st.markdown(f"<div class='rule-card'>{r}</div>", unsafe_allow_html=True)

        # ── Feature Table ────────────────────────────────────────
        st.markdown("<div class='section-head'>🧬 Extracted Biomarkers</div>", unsafe_allow_html=True)

        feat_rows = []
        for i, name in enumerate(feature_names):
            val = feature_vector[i]
            status = classify_feature(name, val)
            rng = FEATURE_NORMAL_RANGES.get(name, None)
            rng_str = f"{rng[0]} – {rng[1]}" if rng else "—"
            css = "feat-normal" if status == "normal" else "feat-abnormal"
            feat_rows.append({
                "Feature": name,
                "Value": f"{val:.6f}",
                "Normal Range": rng_str,
                "Status": f"<span class='{css}'>{'✓ Normal' if status == 'normal' else '⚠ Abnormal'}</span>",
            })

        feat_df = pd.DataFrame(feat_rows)
        st.markdown(feat_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # ── Supplementary Features ───────────────────────────────
        with st.expander("📈 Supplementary Features (MFCCs, Spectral)"):
            supp = feats["supplementary"]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**MFCC Means**")
                mfcc_df = pd.DataFrame({
                    "MFCC": [f"MFCC-{i}" for i in range(len(supp["mfcc_mean"]))],
                    "Value": [f"{v:.4f}" for v in supp["mfcc_mean"]],
                })
                st.dataframe(mfcc_df, hide_index=True)
            with col_b:
                st.markdown("**Spectral Features**")
                for key in ["spectral_centroid", "spectral_bandwidth",
                            "spectral_rolloff", "zero_crossing_rate"]:
                    st.metric(key.replace("_", " ").title(), f"{supp[key]:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # Page: Model Comparison
    # ═══════════════════════════════════════════════════════════════
    elif page.startswith("📊"):
        st.markdown(
            "<h2 class='section-head'>📊 Model Comparison</h2>",
            unsafe_allow_html=True,
        )

        comp_path = os.path.join(MODELS_DIR, "comparison_chart.png")
        if os.path.exists(comp_path):
            st.image(comp_path, caption="10-Fold CV Model Comparison", use_container_width=True)
        else:
            st.info("Run `python train_compare.py` to generate comparison results.")

        imp_path = os.path.join(MODELS_DIR, "rf_feature_importance.png")
        if os.path.exists(imp_path):
            st.image(imp_path, caption="Random Forest — Feature Importance", use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # Page: Explainability
    # ═══════════════════════════════════════════════════════════════
    elif page.startswith("🔍"):
        st.markdown(
            "<h2 class='section-head'>🔍 ANFIS Explainability</h2>",
            unsafe_allow_html=True,
        )

        if not has_models:
            st.error("Train models first with `python train_compare.py`.")
            return

        anfis = models["anfis"]
        sel_idx = models["anfis_sel_idx"]
        sel_names = models.get("anfis_sel_names", [f"f{i}" for i in sel_idx])

        # MF plots
        st.markdown("### Gaussian Membership Functions")
        fig_mf = plot_membership_functions(anfis, feature_names=sel_names)
        st.pyplot(fig_mf, use_container_width=True)

        # Rules
        st.markdown("### Fuzzy Rules (Top 10)")
        rules = extract_fuzzy_rules(anfis, feature_names=sel_names, top_k=10)
        for r in rules:
            st.markdown(f"<div class='rule-card'>{r}</div>", unsafe_allow_html=True)

        # Feature importance
        st.markdown("### Permutation Feature Importance")
        df = load_uci_dataset()
        X, y, fnames = get_feature_matrix(df)
        anfis_scaler = models["anfis_scaler"]

        imp_df = compute_feature_importance(
            anfis, X, y, fnames,
            sel_idx=sel_idx,
            scaler=anfis_scaler,
            n_repeats=5,
        )
        st.dataframe(imp_df, hide_index=True, use_container_width=True)

        # Decision surface
        st.markdown("### Decision Surface (Top 2 Features)")
        fig_ds = plot_decision_surface(
            anfis, X, y,
            feature_pair=(0, 1),
            feature_names=sel_names,
            sel_idx=sel_idx,
            scaler=anfis_scaler,
        )
        st.pyplot(fig_ds, use_container_width=True)


# ─── Entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
