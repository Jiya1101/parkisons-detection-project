"""
Real-time Inference Module for Parkinson's Disease Detection.

Streams audio from the microphone via sounddevice, accumulates 1-second
buffers, extracts features, and runs ANFIS inference with live risk
score updates in the console.

Usage:
    python realtime_inference.py                # default model path
    python realtime_inference.py --model models/anfis_model.pt
"""

import argparse
import os
import sys
import time
import threading
import numpy as np
import torch
import joblib

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[WARN] sounddevice not available — mic streaming disabled.")

from preprocess import reduce_noise, normalize_amplitude, clip_silence
from feature_extraction import extract_features
from anfis_model import ANFIS
from utils.helpers import MODELS_DIR


class RealtimeInference:
    """
    Live microphone → feature extraction → ANFIS prediction loop.

    Attributes:
        model: Trained ANFIS model.
        scaler: StandardScaler fitted on training data.
        sel_idx: Indices of selected features.
        sr: Sample rate (default 22050).
        buffer_sec: Seconds of audio to accumulate before inference.
    """

    def __init__(
        self,
        model_path: str = None,
        sr: int = 22050,
        buffer_sec: float = 1.5,
    ):
        if sd is None:
            raise RuntimeError(
                "sounddevice is required. Install with: pip install sounddevice"
            )

        self.sr = sr
        self.buffer_sec = buffer_sec
        self._running = False
        self._stream = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

        # ── Load model ──────────────────────────────────────────────
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "anfis_model.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run train_compare.py first."
            )

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = ANFIS(
            n_inputs=ckpt["n_inputs"],
            n_mfs=ckpt["n_mfs"],
            n_rules=ckpt["n_rules"],
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.sel_idx = ckpt["sel_idx"]
        self.sel_names = ckpt.get("sel_names", [f"f{i}" for i in self.sel_idx])

        # Reconstruct scaler
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.scaler.mean_ = ckpt["scaler_mean"]
        self.scaler.scale_ = ckpt["scaler_scale"]
        self.scaler.var_ = ckpt["scaler_scale"] ** 2
        self.scaler.n_features_in_ = len(self.sel_idx)

        print(f"[INFO] Model loaded from {model_path}")
        print(f"[INFO] Selected features: {self.sel_names}")

    # ──────────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback — accumulates audio into the buffer."""
        if status:
            print(f"[WARN] Audio status: {status}", file=sys.stderr)
        with self._lock:
            self._buffer = np.append(
                self._buffer, indata[:, 0].astype(np.float32)
            )

    def start_stream(self):
        """Start the microphone stream and inference loop."""
        print("\n" + "=" * 55)
        print("  🎤  Real-Time Parkinson's Risk Monitor")
        print("=" * 55)
        print("  Speak a sustained 'aah' vowel into the microphone.")
        print("  Press Ctrl+C to stop.\n")

        self._running = True
        self._buffer = np.array([], dtype=np.float32)

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()

        try:
            while self._running:
                time.sleep(0.1)
                self._maybe_process()
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
        finally:
            self.stop_stream()

    def stop_stream(self):
        """Stop the microphone stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[INFO] Stream stopped.")

    def _maybe_process(self):
        """Process buffer if it has enough audio."""
        min_samples = int(self.sr * self.buffer_sec)
        with self._lock:
            if len(self._buffer) < min_samples:
                return
            chunk = self._buffer[:min_samples].copy()
            # Keep last half for overlap
            self._buffer = self._buffer[min_samples // 2:]

        risk = self.process_chunk(chunk)
        self._display_risk(risk)

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Run the full pipeline on a single audio chunk.

        Args:
            audio_chunk: 1-D float array of audio samples.

        Returns:
            Risk score ∈ [0, 1].
        """
        t0 = time.time()

        # Preprocess
        audio = reduce_noise(audio_chunk, self.sr)
        audio = clip_silence(audio, self.sr)
        audio = normalize_amplitude(audio)

        # Extract features
        feats = extract_features(audio, sr=self.sr)
        vec = feats["feature_vector"]  # (22,)

        # Select & scale
        vec_sel = vec[self.sel_idx]
        vec_scaled = self.scaler.transform(vec_sel.reshape(1, -1))

        # Inference
        with torch.no_grad():
            x_t = torch.tensor(vec_scaled, dtype=torch.float32)
            risk = self.model(x_t).item()

        dt = (time.time() - t0) * 1000
        if dt > 100:
            print(f"  [WARN] Inference took {dt:.0f} ms (target <100 ms)")

        return risk

    @staticmethod
    def _display_risk(risk: float):
        """Pretty-print the risk score to the console."""
        bar_len = 30
        filled = int(risk * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        if risk < 0.3:
            label, color = "LOW", "\033[92m"
        elif risk < 0.6:
            label, color = "MODERATE", "\033[93m"
        elif risk < 0.8:
            label, color = "ELEVATED", "\033[33m"
        else:
            label, color = "HIGH", "\033[91m"
        reset = "\033[0m"

        print(
            f"  {color}Risk: {risk:.1%}  [{bar}]  {label}{reset}",
            end="\r",
        )


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Parkinson's detection from microphone."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to ANFIS model checkpoint (.pt)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate (default 22050)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=1.5,
        help="Buffer length in seconds (default 1.5)",
    )
    args = parser.parse_args()

    rt = RealtimeInference(
        model_path=args.model,
        sr=args.sr,
        buffer_sec=args.buffer,
    )
    rt.start_stream()
