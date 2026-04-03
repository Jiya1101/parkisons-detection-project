"""
Feature Extraction Module for Parkinson's Disease Detection.

Extracts 22-dimensional biomarker feature vectors from sustained vowel
recordings.  Features are aligned with the UCI Parkinson's dataset schema
so that models trained on UCI data can score new audio inputs.

Feature groups
──────────────
  Frequency perturbation  : Jitter (local, abs, RAP, PPQ5, DDP)          ×5
  Amplitude perturbation  : Shimmer (local, dB, APQ3, APQ5, APQ11, DDA)  ×6
  Fundamental frequency   : F0 mean, max, min (+ std reused as PPE dim)  ×3
  Harmonics               : HNR, NHR                                     ×2
  Nonlinear dynamics      : RPDE, DFA, spread1, spread2, D2, PPE         ×6
                                                                   Total  22

Supplementary features (MFCCs, spectral centroid, ZCR, etc.) are also
extracted for UI display but are *not* part of the core 22-dim vector.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist


# ─── Main Entry Point ───────────────────────────────────────────────────────


def extract_features(wav_path_or_audio, sr: int = 22050) -> dict:
    """
    Extract all features from an audio file or array.

    Args:
        wav_path_or_audio: Path to a WAV file (str) or a 1-D float audio
                           array.  When an array is passed, `sr` is used.
        sr: Sample rate (used when wav_path_or_audio is an array).

    Returns:
        dict with keys:
            'feature_vector' – np.ndarray of shape (22,)
            'feature_names'  – list of 22 UCI feature names
            'supplementary'  – dict of extra features (MFCCs, spectral, etc.)
            'raw'            – dict of individual group dicts
    """
    # Resolve audio
    if isinstance(wav_path_or_audio, str):
        audio, sr = librosa.load(wav_path_or_audio, sr=sr, mono=True)
        wav_path = wav_path_or_audio
    else:
        audio = np.asarray(wav_path_or_audio, dtype=np.float64)
        wav_path = None

    # -- Core 22 features ------------------------------------------------
    jitter = extract_jitter_shimmer_praat(audio, sr, kind="jitter")
    shimmer = extract_jitter_shimmer_praat(audio, sr, kind="shimmer")
    f0_feats = extract_f0_features(audio, sr)
    hnr_feats = extract_hnr_nhr(audio, sr)
    nonlinear = extract_nonlinear_features(audio, sr)

    # Assemble in UCI column order
    feature_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
        "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
        "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
    ]

    vec = np.array([
        f0_feats["f0_mean"],         # MDVP:Fo(Hz)
        f0_feats["f0_max"],          # MDVP:Fhi(Hz)
        f0_feats["f0_min"],          # MDVP:Flo(Hz)
        jitter["jitter_local"],      # MDVP:Jitter(%)
        jitter["jitter_abs"],        # MDVP:Jitter(Abs)
        jitter["jitter_rap"],        # MDVP:RAP
        jitter["jitter_ppq5"],       # MDVP:PPQ
        jitter["jitter_ddp"],        # Jitter:DDP
        shimmer["shimmer_local"],    # MDVP:Shimmer
        shimmer["shimmer_db"],       # MDVP:Shimmer(dB)
        shimmer["shimmer_apq3"],     # Shimmer:APQ3
        shimmer["shimmer_apq5"],     # Shimmer:APQ5
        shimmer["shimmer_apq11"],    # MDVP:APQ
        shimmer["shimmer_dda"],      # Shimmer:DDA
        hnr_feats["nhr"],            # NHR
        hnr_feats["hnr"],            # HNR
        nonlinear["rpde"],           # RPDE
        nonlinear["dfa"],            # DFA
        nonlinear["spread1"],        # spread1
        nonlinear["spread2"],        # spread2
        nonlinear["d2"],             # D2
        nonlinear["ppe"],            # PPE
    ], dtype=np.float64)

    # Replace NaN / Inf with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # -- Supplementary features -------------------------------------------
    supplementary = {}
    supplementary.update(extract_mfcc_features(audio, sr))
    supplementary.update(extract_spectral_features(audio, sr))

    raw = {
        "jitter": jitter,
        "shimmer": shimmer,
        "f0": f0_feats,
        "hnr": hnr_feats,
        "nonlinear": nonlinear,
    }

    return {
        "feature_vector": vec,
        "feature_names": feature_names,
        "supplementary": supplementary,
        "raw": raw,
    }


# ─── Jitter & Shimmer (via Parselmouth / Praat) ─────────────────────────────


def extract_jitter_shimmer_praat(
    audio: np.ndarray, sr: int, kind: str = "jitter"
) -> dict:
    """
    Extract jitter or shimmer features using Praat (via parselmouth).

    Args:
        audio: Audio signal.
        sr: Sample rate.
        kind: 'jitter' or 'shimmer'.

    Returns:
        dict of feature values.
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    # Pitch range for voiced speech
    f0_min, f0_max = 75, 600

    try:
        point_process = call(
            sound, "To PointProcess (periodic, cc)", f0_min, f0_max
        )
    except Exception:
        # Return zeros if Praat cannot find pitch periods
        if kind == "jitter":
            return {k: 0.0 for k in [
                "jitter_local", "jitter_abs", "jitter_rap",
                "jitter_ppq5", "jitter_ddp",
            ]}
        else:
            return {k: 0.0 for k in [
                "shimmer_local", "shimmer_db", "shimmer_apq3",
                "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
            ]}

    if kind == "jitter":
        return _extract_jitter(point_process)
    else:
        return _extract_shimmer(sound, point_process)


def _extract_jitter(pp) -> dict:
    """Extract five jitter measures from a Praat PointProcess."""
    def _safe(fn, *a, **kw):
        try:
            v = fn(*a, **kw)
            return 0.0 if (v is None or np.isnan(v)) else float(v)
        except Exception:
            return 0.0

    return {
        "jitter_local": _safe(
            call, pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        ),
        "jitter_abs": _safe(
            call, pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
        ),
        "jitter_rap": _safe(
            call, pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
        ),
        "jitter_ppq5": _safe(
            call, pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
        ),
        "jitter_ddp": _safe(
            call, pp, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3
        ),
    }


def _extract_shimmer(sound, pp) -> dict:
    """Extract six shimmer measures from a Praat Sound + PointProcess."""
    def _safe(fn, *a, **kw):
        try:
            v = fn(*a, **kw)
            return 0.0 if (v is None or np.isnan(v)) else float(v)
        except Exception:
            return 0.0

    return {
        "shimmer_local": _safe(
            call, [sound, pp], "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        ),
        "shimmer_db": _safe(
            call, [sound, pp], "Get shimmer (local_dB)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        ),
        "shimmer_apq3": _safe(
            call, [sound, pp], "Get shimmer (apq3)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        ),
        "shimmer_apq5": _safe(
            call, [sound, pp], "Get shimmer (apq5)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        ),
        "shimmer_apq11": _safe(
            call, [sound, pp], "Get shimmer (apq11)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        ),
        "shimmer_dda": _safe(
            call, [sound, pp], "Get shimmer (dda)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        ),
    }


# ─── F0 Features ────────────────────────────────────────────────────────────


def extract_f0_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract fundamental frequency (F0) statistics via Praat.

    Returns:
        dict with f0_mean, f0_max, f0_min, f0_std.
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    except Exception:
        f0_mean = f0_std = f0_min = f0_max = 0.0

    def _s(v):
        return 0.0 if (v is None or np.isnan(v)) else float(v)

    return {
        "f0_mean": _s(f0_mean),
        "f0_max": _s(f0_max),
        "f0_min": _s(f0_min),
        "f0_std": _s(f0_std),
    }


# ─── HNR / NHR ──────────────────────────────────────────────────────────────


def extract_hnr_nhr(audio: np.ndarray, sr: int) -> dict:
    """
    Extract Harmonics-to-Noise Ratio (HNR) and Noise-to-Harmonics Ratio
    (NHR) via Praat.
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    try:
        harmonicity = call(
            sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
        )
        hnr = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr = 0.0

    if hnr is None or np.isnan(hnr):
        hnr = 0.0

    hnr = float(hnr)
    nhr = 1.0 / hnr if hnr > 0 else 0.0

    return {"hnr": hnr, "nhr": nhr}


# ─── Nonlinear Features ─────────────────────────────────────────────────────


def extract_nonlinear_features(audio: np.ndarray, sr: int) -> dict:
    """
    Compute nonlinear-dynamics features: RPDE, DFA, D2, spread1,
    spread2, and PPE.  These are computed from the audio waveform
    (RPDE, DFA, D2) and from the F0 contour (spread1, spread2, PPE).
    """
    # Downsample for computational efficiency
    target_len = 4000
    if len(audio) > target_len:
        step = len(audio) // target_len
        signal = audio[::step][:target_len]
    else:
        signal = audio

    rpde = _compute_rpde(signal)
    dfa = _compute_dfa(signal)
    d2 = _compute_d2(signal)

    # F0 contour for spread1, spread2, PPE
    f0_contour = _extract_f0_contour(audio, sr)
    spread1, spread2 = _compute_spread(f0_contour)
    ppe = _compute_ppe(f0_contour)

    return {
        "rpde": rpde,
        "dfa": dfa,
        "d2": d2,
        "spread1": spread1,
        "spread2": spread2,
        "ppe": ppe,
    }


def _extract_f0_contour(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return the F0 contour (Hz) via Praat; zeros for unvoiced frames."""
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        n_frames = call(pitch, "Get number of frames")
        f0 = np.array([
            call(pitch, "Get value in frame", i + 1, "Hertz")
            for i in range(int(n_frames))
        ])
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        f0 = np.zeros(1)
    return f0


# -- RPDE -----------------------------------------------------------------

def _compute_rpde(
    signal: np.ndarray,
    m: int = 10,
    tau: int = 1,
    epsilon: float = None,
    max_recurrence: int = 500,
) -> float:
    """
    Recurrence Period Density Entropy.

    1. Time-delay embed the signal.
    2. For each embedded vector find neighbours within *epsilon*.
    3. Track recurrence times (distances in index space between revisits).
    4. Compute normalised Shannon entropy of the recurrence-time histogram.
    """
    N = len(signal)
    n_vectors = N - (m - 1) * tau
    if n_vectors < 20:
        return 0.0

    embedded = np.array(
        [signal[i: i + m * tau: tau] for i in range(n_vectors)]
    )

    if epsilon is None:
        epsilon = 0.12 * np.std(embedded)
    if epsilon < 1e-10:
        return 0.0

    tree = KDTree(embedded)

    recurrence_times = []
    # Sample a subset for speed
    sample_idx = np.random.choice(
        n_vectors, size=min(n_vectors, 500), replace=False
    )
    for i in sample_idx:
        neighbours = tree.query_ball_point(embedded[i], epsilon)
        for j in neighbours:
            if j > i:
                rt = j - i
                if rt <= max_recurrence:
                    recurrence_times.append(rt)

    if len(recurrence_times) == 0:
        return 0.0

    counts = np.bincount(
        np.array(recurrence_times), minlength=max_recurrence + 1
    )[1:]
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(max_recurrence)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# -- DFA ------------------------------------------------------------------

def _compute_dfa(
    signal: np.ndarray,
    fit_order: int = 1,
    min_win: int = 10,
    max_win: int = None,
) -> float:
    """
    Detrended Fluctuation Analysis exponent *α*.

    α ≈ 0.5 → white noise  |  α ≈ 1.0 → 1/f noise  |  α ≈ 1.5 → brown noise
    """
    N = len(signal)
    if N < 30:
        return 0.5

    if max_win is None:
        max_win = N // 4

    # Cumulative sum of demeaned series
    y = np.cumsum(signal - np.mean(signal))

    window_sizes = np.unique(
        np.logspace(np.log10(min_win), np.log10(max(max_win, min_win + 1)), 20)
        .astype(int)
    )
    window_sizes = window_sizes[window_sizes >= 4]

    fluctuations = []
    valid_sizes = []
    for n in window_sizes:
        n_windows = N // n
        if n_windows < 1:
            continue
        rms_list = []
        for w in range(n_windows):
            seg = y[w * n: (w + 1) * n]
            x_ax = np.arange(n)
            coeffs = np.polyfit(x_ax, seg, fit_order)
            trend = np.polyval(coeffs, x_ax)
            residual = seg - trend
            rms_list.append(np.sqrt(np.mean(residual ** 2)))
        fluctuations.append(np.mean(rms_list))
        valid_sizes.append(n)

    if len(valid_sizes) < 3:
        return 0.5

    log_n = np.log(np.array(valid_sizes, dtype=float))
    log_f = np.log(np.array(fluctuations, dtype=float) + 1e-12)

    alpha, _ = np.polyfit(log_n, log_f, 1)
    return float(alpha)


# -- D2 (Correlation Dimension) ------------------------------------------

def _compute_d2(
    signal: np.ndarray,
    m: int = 10,
    tau: int = 1,
    n_r: int = 20,
) -> float:
    """
    Grassberger–Procaccia correlation dimension D₂.

    Uses time-delay embedding and log–log regression of the
    correlation integral C(r).
    """
    N = len(signal)
    n_vectors = N - (m - 1) * tau
    if n_vectors < 20:
        return 2.0

    embedded = np.array(
        [signal[i: i + m * tau: tau] for i in range(n_vectors)]
    )

    # Subsample to keep computation tractable
    max_pts = 600
    if n_vectors > max_pts:
        idx = np.random.choice(n_vectors, max_pts, replace=False)
        embedded = embedded[idx]

    distances = pdist(embedded)
    if len(distances) == 0:
        return 2.0

    # Non-zero distances only
    distances = distances[distances > 0]
    if len(distances) == 0:
        return 2.0

    r_min = np.percentile(distances, 2)
    r_max = np.percentile(distances, 98)
    if r_min >= r_max or r_min <= 0:
        return 2.0

    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_r)

    n_pairs = len(distances)
    C = np.array([np.sum(distances < r) / n_pairs for r in r_values])

    valid = C > 0
    log_r = np.log(r_values[valid])
    log_C = np.log(C[valid])

    if len(log_r) < 4:
        return 2.0

    # Fit the middle portion of the scaling region
    n = len(log_r)
    start, end = n // 4, 3 * n // 4
    if end - start < 3:
        start, end = 0, n

    d2, _ = np.polyfit(log_r[start:end], log_C[start:end], 1)
    return float(d2)


# -- Spread1 / Spread2 ---------------------------------------------------

def _compute_spread(f0: np.ndarray):
    """
    Nonlinear measures of fundamental frequency variation.

    spread1 = std of log(F0)
    spread2 = mean |Δlog(F0)|
    """
    voiced = f0[f0 > 0]
    if len(voiced) < 2:
        return 0.0, 0.0

    log_f0 = np.log(voiced + 1e-12)
    spread1 = float(np.std(log_f0))
    spread2 = float(np.mean(np.abs(np.diff(log_f0))))
    return spread1, spread2


# -- PPE ------------------------------------------------------------------

def _compute_ppe(f0: np.ndarray, n_semitones: int = 12) -> float:
    """
    Pitch Period Entropy.

    1. Convert F0 contour to semitone scale relative to its mean.
    2. Round to nearest semitone.
    3. Return Shannon entropy of the semitone histogram.
    """
    voiced = f0[f0 > 0]
    if len(voiced) < 2:
        return 0.0

    f0_ref = np.mean(voiced)
    semitones = 12.0 * np.log2(voiced / f0_ref + 1e-12)
    semitones_rounded = np.round(semitones).astype(int)

    unique, counts = np.unique(semitones_rounded, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


# ─── Supplementary Features (for UI display) ────────────────────────────────


def extract_mfcc_features(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """Extract MFCCs and their deltas."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    return {
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_delta_mean": mfcc_delta_mean.tolist(),
    }


def extract_spectral_features(audio: np.ndarray, sr: int) -> dict:
    """Extract spectral centroid, bandwidth, rolloff, and ZCR."""
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr,
    }
