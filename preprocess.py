"""
Audio Preprocessing Module for Parkinson's Disease Detection.

Handles noise reduction, amplitude normalization, silence clipping,
and windowing of sustained vowel "aah" recordings.
"""

import numpy as np
import librosa
import noisereduce as nr


def _finalize_preprocessed(audio: np.ndarray, sr: int, original_audio: np.ndarray) -> dict:
    windows = segment_windows(audio, sr, win_ms=25, hop_ms=10)
    duration = len(audio) / sr
    return {
        "audio": audio,
        "sr": sr,
        "duration": duration,
        "windows": windows,
        "original_audio": original_audio,
    }


def load_and_preprocess(wav_path: str, sr: int = 22050) -> dict:
    """
    Load and fully preprocess an audio file.

    Pipeline:
        1. Load audio (mono, resampled to target SR)
        2. Spectral-gating noise reduction
        3. Silence clipping using energy-based VAD
        4. Amplitude normalization to [-1, 1]
        5. Hamming windowing (25ms frames, 10ms hop)

    Args:
        wav_path: Path to .wav audio file.
        sr: Target sample rate (default 22050 Hz).

    Returns:
        dict with keys:
            'audio'          – preprocessed audio signal (np.ndarray)
            'sr'             – sample rate (int)
            'duration'       – duration in seconds (float)
            'windows'        – windowed frames, shape (N, win_length) (np.ndarray)
            'original_audio' – unprocessed loaded audio (np.ndarray)
    """
    # Load audio
    audio, sr = librosa.load(wav_path, sr=sr, mono=True)
    original_audio = audio.copy()

    # Step 1: Noise reduction
    audio = reduce_noise(audio, sr)

    # Step 2: Silence clipping
    audio = clip_silence(audio, sr, threshold_db=-40)

    # Step 3: Amplitude normalization
    audio = normalize_amplitude(audio)

    # Step 4: Keep the most stable voiced vowel segment
    audio = select_stable_voiced_segment(audio, sr, target_sec=2.5)

    # Step 5: Windowing
    return _finalize_preprocessed(audio, sr, original_audio)


def load_and_preprocess_bytes(audio_bytes: bytes, sr: int = 22050) -> dict:
    """
    Preprocess raw audio bytes (e.g. from Streamlit mic recorder).

    Args:
        audio_bytes: Raw WAV bytes.
        sr: Target sample rate.

    Returns:
        Same dict as load_and_preprocess().
    """
    import io
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    audio, orig_sr = sf.read(buf, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    original_audio = audio.copy()

    audio = reduce_noise(audio, sr)
    audio = clip_silence(audio, sr, threshold_db=-40)
    audio = normalize_amplitude(audio)
    audio = select_stable_voiced_segment(audio, sr, target_sec=2.5)
    return _finalize_preprocessed(audio, sr, original_audio)


# ─── Individual Processing Steps ────────────────────────────────────────────


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply spectral-gating noise reduction.

    Uses the noisereduce library's stationary noise reduction, which
    estimates a noise profile from the quieter parts of the signal and
    gates out noise below that profile.

    Args:
        audio: Audio signal (1-D float array).
        sr: Sample rate.

    Returns:
        Noise-reduced audio signal.
    """
    reduced = nr.reduce_noise(
        y=audio, sr=sr, prop_decrease=0.8, stationary=True
    )
    return reduced


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1] range (peak normalization).

    Args:
        audio: Audio signal.

    Returns:
        Peak-normalized audio signal.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def clip_silence(
    audio: np.ndarray, sr: int, threshold_db: float = -40
) -> np.ndarray:
    """
    Remove silence from the beginning and end of the audio using
    energy-based Voice Activity Detection (VAD).

    Args:
        audio: Audio signal.
        sr: Sample rate.
        threshold_db: Energy threshold in dB below peak; frames quieter
                      than this are considered silent.

    Returns:
        Audio with silence trimmed from leading/trailing edges.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))

    # Safety: keep original if too much was trimmed (<100 ms remaining)
    if len(trimmed) < int(sr * 0.1):
        return audio

    return trimmed


def select_stable_voiced_segment(
    audio: np.ndarray,
    sr: int,
    target_sec: float = 2.5,
    frame_ms: int = 40,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Select the most stable voiced region from a sustained-vowel recording.

    We score short frames using a mix of energy, low zero-crossing rate,
    and pitch stability, then keep the best contiguous segment near target_sec.
    """
    target_len = int(sr * target_sec)
    if len(audio) <= target_len:
        return audio

    frame_length = max(int(sr * frame_ms / 1000), 256)
    hop_length = max(int(sr * hop_ms / 1000), 64)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    try:
        f0 = librosa.yin(
            audio,
            fmin=75.0,
            fmax=500.0,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        voiced = np.isfinite(f0) & (f0 > 0)
    except Exception:
        f0 = np.zeros_like(rms)
        voiced = np.zeros_like(rms, dtype=bool)

    if len(rms) < 3:
        start = max((len(audio) - target_len) // 2, 0)
        return audio[start:start + target_len]

    rms_norm = rms / (np.max(rms) + 1e-8)
    zcr_norm = zcr / (np.max(zcr) + 1e-8)

    f0_stability = np.zeros_like(rms_norm)
    if voiced.any():
        f0_safe = np.where(voiced, f0, np.nan)
        for i in range(len(f0_safe)):
            lo = max(0, i - 2)
            hi = min(len(f0_safe), i + 3)
            local = f0_safe[lo:hi]
            if np.sum(np.isfinite(local)) >= 2:
                mean_local = np.nanmean(local)
                std_local = np.nanstd(local)
                f0_stability[i] = 1.0 / (1.0 + std_local / (mean_local + 1e-6))

    frame_score = (
        0.55 * rms_norm +
        0.25 * (1.0 - np.clip(zcr_norm, 0.0, 1.0)) +
        0.20 * f0_stability
    )
    frame_score = np.where(voiced, frame_score + 0.15, frame_score - 0.25)

    seg_frames = max(int(np.ceil((target_len - frame_length) / hop_length)) + 1, 1)
    if len(frame_score) <= seg_frames:
        return audio

    best_idx = 0
    best_score = -np.inf
    for start_f in range(0, len(frame_score) - seg_frames + 1):
        end_f = start_f + seg_frames
        window_scores = frame_score[start_f:end_f]
        voiced_ratio = float(np.mean(voiced[start_f:end_f])) if voiced.any() else 0.0
        score = float(np.mean(window_scores)) + 0.35 * voiced_ratio
        if score > best_score:
            best_score = score
            best_idx = start_f

    start = best_idx * hop_length
    end = min(start + target_len, len(audio))
    segment = audio[start:end]

    if len(segment) < int(sr * 1.0):
        return audio
    return segment


def segment_windows(
    audio: np.ndarray,
    sr: int,
    win_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Segment audio into overlapping Hamming-windowed frames.

    Args:
        audio: Audio signal.
        sr: Sample rate.
        win_ms: Window length in milliseconds.
        hop_ms: Hop length in milliseconds.

    Returns:
        2-D array of shape (n_windows, window_length).
    """
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    # Pad if audio is shorter than one window
    if len(audio) < win_length:
        audio = np.pad(audio, (0, win_length - len(audio)))

    frames = librosa.util.frame(
        audio, frame_length=win_length, hop_length=hop_length
    ).T  # (n_windows, win_length)

    hamming = np.hamming(win_length)
    windowed_frames = frames * hamming[np.newaxis, :]

    return windowed_frames
