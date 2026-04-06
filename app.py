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
from anfis_model import (
    ANFIS,
    apply_output_calibration,
    score_input_reliability,
    temper_risk_by_reliability,
)
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
        models["anfis_calibration"] = {
            "coef": ckpt.get("calibration_coef", 1.0),
            "intercept": ckpt.get("calibration_intercept", 0.0),
        }
        if "feature_p01" in ckpt and "feature_p99" in ckpt:
            models["anfis_feature_profile"] = {
                "p01": ckpt["feature_p01"],
                "p99": ckpt["feature_p99"],
                "abs_z_p95": ckpt.get("feature_abs_z_p95", 2.5),
                "abs_z_p99": ckpt.get("feature_abs_z_p99", 4.0),
            }

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


def _score_anfis_sample(anfis, vec_scaled, calibration=None, feature_profile=None):
    with torch.no_grad():
        raw_risk = anfis(torch.tensor(vec_scaled, dtype=torch.float32)).item()

    calibrated_risk = float(apply_output_calibration(raw_risk, calibration))
    reliability = score_input_reliability(vec_scaled[0], feature_profile)
    displayed_risk = float(
        temper_risk_by_reliability(calibrated_risk, reliability["reliability"])
    )
    return {
        "raw_risk": raw_risk,
        "calibrated_risk": calibrated_risk,
        "displayed_risk": displayed_risk,
        **reliability,
    }


def _confidence_state(score_info):
    rel = score_info["reliability"]
    if rel < 0.45:
        return "low", "Insufficient Confidence", "risk-mod"
    if rel < 0.70:
        return "moderate", "Low Confidence", "risk-mod"
    label, css = _risk_label(score_info["displayed_risk"])
    return "high", label, css


def _recording_quality_summary(prep, feature_vector, feature_names, sel_idx, score_info):
    duration = float(prep["duration"])
    duration_ok = 1.5 <= duration <= 3.5

    abnormal_count = 0
    for i, name in enumerate(feature_names):
        if classify_feature(name, float(feature_vector[i])) == "abnormal":
            abnormal_count += 1

    selected_feature_names = [feature_names[i] for i in sel_idx]
    selected_abnormal = 0
    for i in sel_idx:
        if classify_feature(feature_names[i], float(feature_vector[i])) == "abnormal":
            selected_abnormal += 1

    rel = float(score_info["reliability"])
    outside_count = int(score_info["n_outside"])
    max_abs_z = float(score_info["max_abs_z"])

    if rel < 0.45:
        quality_label = "Poor"
        quality_color = "risk-high"
    elif rel < 0.70 or not duration_ok:
        quality_label = "Fair"
        quality_color = "risk-mod"
    else:
        quality_label = "Good"
        quality_color = "risk-low"

    reasons = []
    if not duration_ok:
        reasons.append("Recording length is outside the ideal 1.5-3.5s window.")
    if outside_count > max(2, len(sel_idx) // 4):
        reasons.append("Too many selected features fall outside the training range.")
    if max_abs_z > 3.5:
        reasons.append("The recording is unusually far from the model's scaled feature distribution.")
    if selected_abnormal > max(3, len(sel_idx) // 3):
        reasons.append("Many of the most important voice biomarkers are flagged as abnormal.")
    if not reasons:
        reasons.append("The recording shape is reasonably close to the training distribution.")

    return {
        "quality_label": quality_label,
        "quality_color": quality_color,
        "duration_ok": duration_ok,
        "abnormal_count": abnormal_count,
        "selected_abnormal": selected_abnormal,
        "selected_total": len(sel_idx),
        "outside_count": outside_count,
        "max_abs_z": max_abs_z,
        "reasons": reasons,
        "selected_feature_names": selected_feature_names,
    }


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
        calibration = models.get("anfis_calibration")
        feature_profile = models.get("anfis_feature_profile")

        vec_sel = feature_vector[sel_idx]
        vec_scaled = anfis_scaler.transform(vec_sel.reshape(1, -1))
        score_info = _score_anfis_sample(
            anfis,
            vec_scaled,
            calibration=calibration,
            feature_profile=feature_profile,
        )
        quality_info = _recording_quality_summary(
            prep,
            feature_vector,
            feature_names,
            sel_idx,
            score_info,
        )
        confidence_level, assessment_text, assessment_css = _confidence_state(score_info)
        risk_score = score_info["displayed_risk"] if confidence_level != "low" else 0.50

        # ── KPI Row ──────────────────────────────────────────────
        st.markdown(
            f"""
            <div class='kpi-row'>
                <div class='kpi-card'>
                    <div class='value'>{risk_score:.1%}</div>
                    <div class='label'>Confidence-Aware Risk</div>
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

        rel = score_info["reliability"]
        if rel < 0.45:
            st.warning(
                "This recording is outside the model's reliable operating range. "
                "The gauge is being held at neutral instead of showing a misleading risk."
            )
            st.markdown(
                "- Use a steady sustained `aah` for 2-3 seconds\n"
                "- Keep the microphone distance fixed\n"
                "- Record in a quiet room with minimal echo\n"
                "- Avoid whispering, shouting, or changing pitch mid-recording"
            )
        elif rel < 0.70:
            st.info(
                "Confidence is moderate: some selected voice features are outside the model's "
                "usual training range, so treat this as a screening estimate rather than a probability."
            )

        st.caption(
            f"Raw ANFIS score: {score_info['raw_risk']:.1%} | "
            f"Calibrated score: {score_info['calibrated_risk']:.1%} | "
            f"Reliability: {rel:.0%} | "
            f"Out-of-range selected features: {score_info['n_outside']}/{len(sel_idx)}"
        )

        # ── Gauge Chart ──────────────────────────────────────────
        col_gauge, col_rules = st.columns([1, 1])

        with col_gauge:
            st.markdown("<div class='section-head'>🎯 Risk Gauge</div>", unsafe_allow_html=True)
            gauge_title = (
                "Recording Confidence Too Low"
                if confidence_level == "low"
                else "Confidence-Aware Risk Score"
            )
            fig_gauge = create_gauge_chart(risk_score, title=gauge_title)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_rules:
            st.markdown("<div class='section-head'>📜 Fuzzy Rules Fired</div>", unsafe_allow_html=True)
            sel_names = models.get("anfis_sel_names", [f"f{i}" for i in sel_idx])
            rules = extract_fuzzy_rules(anfis, feature_names=sel_names, top_k=8)
            for r in rules:
                st.markdown(f"<div class='rule-card'>{r}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Recording Quality</div>", unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Quality", quality_info["quality_label"])
        with q2:
            st.metric("Reliability", f"{score_info['reliability']:.0%}")
        with q3:
            st.metric("Outside Range", f"{quality_info['outside_count']}/{len(sel_idx)}")
        with q4:
            st.metric("Max |Z|", f"{quality_info['max_abs_z']:.2f}")

        st.markdown(
            f"Assessment: <span class='{quality_info['quality_color']}'>{quality_info['quality_label']}</span>",
            unsafe_allow_html=True,
        )
        for reason in quality_info["reasons"]:
            st.markdown(f"- {reason}")

        with st.expander("Why this recording was judged this way"):
            st.markdown(
                f"- Audio duration: `{prep['duration']:.2f}s` "
                f"({'ideal' if quality_info['duration_ok'] else 'outside ideal range'})\n"
                f"- Selected features outside training range: `{quality_info['outside_count']}/{len(sel_idx)}`\n"
                f"- Abnormal biomarkers overall: `{quality_info['abnormal_count']}/{len(feature_names)}`\n"
                f"- Abnormal biomarkers among selected inputs: "
                f"`{quality_info['selected_abnormal']}/{quality_info['selected_total']}`"
            )

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
