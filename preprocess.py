"""
Audio Preprocessing Module for Parkinson's Disease Detection.

Handles noise reduction, amplitude normalization, silence clipping,
and windowing of sustained vowel "aah" recordings.
"""

import numpy as np
import librosa
import noisereduce as nr


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

    # Step 4: Windowing
    windows = segment_windows(audio, sr, win_ms=25, hop_ms=10)

    duration = len(audio) / sr

    return {
        "audio": audio,
        "sr": sr,
        "duration": duration,
        "windows": windows,
        "original_audio": original_audio,
    }


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
    windows = segment_windows(audio, sr, win_ms=25, hop_ms=10)

    duration = len(audio) / sr

    return {
        "audio": audio,
        "sr": sr,
        "duration": duration,
        "windows": windows,
        "original_audio": original_audio,
    }


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
