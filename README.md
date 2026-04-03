# Parkinson's Disease Early Detection System

> **Voice-analysis–based screening powered by ANFIS (Adaptive Neuro-Fuzzy Inference System)**

A hybrid soft-computing pipeline that analyses sustained vowel ("aah") recordings to assess Parkinson's disease risk.  The system extracts 22 clinically-validated biomarker features from voice signals and feeds them through a custom PyTorch ANFIS model — with SVM, Random Forest, and Neural Network baselines for comparison.

---

## Architecture

```
  Audio (.wav / mic)
        │
        ▼
  ┌──────────────┐
  │  preprocess   │  Noise reduction · silence clipping · normalisation · windowing
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  feature      │  22-dim biomarker vector  (jitter, shimmer, F0, HNR, RPDE,
  │  extraction   │  DFA, D2, spread1/2, PPE)  +  supplementary MFCCs / spectral
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  ANFIS model  │  Gaussian MFs → rules → TSK consequent → risk score ∈ [0, 1]
  │  (PyTorch)    │  Hybrid training (LSE + backprop) + PSO initialisation
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Streamlit UI │  Gauge · feature table · fuzzy rules · model comparison
  └──────────────┘
```

---

## Project Structure

```
Soft computing/
├── preprocess.py              # Audio preprocessing pipeline
├── feature_extraction.py      # 22-dim biomarker feature extraction
├── anfis_model.py             # Custom PyTorch ANFIS + PSO optimisation
├── train_compare.py           # 10-fold CV training + model comparison
├── explainability.py          # Fuzzy rules, feature importance, MF plots
├── realtime_inference.py      # Live microphone → ANFIS inference
├── app.py                     # Streamlit web UI
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── parkinsons.data        # UCI dataset (auto-downloaded)
├── models/
│   └── *.pt / *.pkl           # Saved model checkpoints
└── utils/
    └── helpers.py             # Shared utilities
```

---

## Installation

```bash
# Clone / navigate to the project directory
cd "c:\Users\yashi\OneDrive\Documents\Soft computing"

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### System Dependencies

| Library | Notes |
|---------|-------|
| `sounddevice` | Requires PortAudio. On Windows the pip wheel includes it. |
| `parselmouth` | Python binding for Praat. Installs via pip. |

---

## Usage

### 1. Train Models

```bash
python train_compare.py
```

This will:
- Auto-download the UCI Parkinson's dataset (195 samples)
- Run 10-fold stratified cross-validation on **ANFIS**, **SVM**, **Random Forest**, and **Neural Network**
- Train final models on the full dataset
- Save checkpoints to `models/`

### 2. Streamlit Web App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.  Upload a `.wav` file or record directly from the microphone.

### 3. Real-time Inference (CLI)

```bash
python realtime_inference.py
```

Speak a sustained "aah" vowel.  The system prints live risk scores to the terminal.

---

## Feature Groups

| Group | Features | Count |
|-------|----------|-------|
| Frequency perturbation | Jitter (local, abs, RAP, PPQ5, DDP) | 5 |
| Amplitude perturbation | Shimmer (local, dB, APQ3, APQ5, APQ11, DDA) | 6 |
| Fundamental frequency | F0 mean, max, min | 3 |
| Harmonics | HNR, NHR | 2 |
| Nonlinear dynamics | RPDE, DFA, spread1, spread2, D2, PPE | 6 |
| **Total** | | **22** |

---

## ANFIS Architecture

The ANFIS (Takagi-Sugeno Type-1) model consists of five layers:

1. **Fuzzification** — Gaussian membership functions with learnable (μ, σ) for each input × MF
2. **Rule Firing** — Product T-norm across selected MFs
3. **Normalisation** — Normalised firing strengths
4. **Consequent** — Linear TSK: fₖ = p₀ + Σpᵢxᵢ
5. **Defuzzification** — Weighted sum → sigmoid → risk ∈ [0, 1]

### Training Strategy

1. **Feature Selection** — Top 8 features by mutual information
2. **PSO Initialisation** — Particle Swarm Optimisation for MF parameters
3. **Hybrid Learning** — LSE (consequent) + gradient descent (premise) per epoch
4. **Rule Pruning** — Remove rules with < 0.5% average firing strength
5. **Fine-tuning** — Additional training after pruning

With 8 inputs × 2 MFs → 256 initial rules → pruned to ~16–30 active rules.

---

## Benchmark Results

Target metrics (UCI dataset, 10-fold CV):

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 92% |
| Sensitivity (Recall) | ≥ 90% |
| Specificity | ≥ 88% |
| F1-Score | ≥ 90% |
| AUC-ROC | ≥ 0.95 |

---

## References

1. Little, M.A. et al. (2007) "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection." *BioMedical Engineering OnLine*.
2. UCI Machine Learning Repository — [Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
3. Jang, J.-S.R. (1993) "ANFIS: Adaptive-Network-Based Fuzzy Inference System." *IEEE Trans. SMC*.

---

## License

Academic project — Soft Computing course.
