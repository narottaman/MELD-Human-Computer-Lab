"""Deterministic non-learned audio and visual features from MELD MP4 clips."""
from pathlib import Path
import subprocess
import numpy as np
import cv2
from scipy.io import wavfile

AUDIO_DIM = 12
VISION_DIM = 12
FEATURE_DIM = AUDIO_DIM + VISION_DIM + 2


def audio_features(path):
    if not path or not Path(path).is_file():
        return np.zeros(AUDIO_DIM, np.float32), 0.0
    try:
        proc = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vn', '-ac', '1', '-ar', '16000', '-f', 'wav', '-'], capture_output=True, timeout=35)
        if proc.returncode or not proc.stdout:
            raise RuntimeError(proc.stderr.decode(errors='ignore')[:200])
        import io
        sr, raw = wavfile.read(io.BytesIO(proc.stdout))
        x = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            x /= max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
        if len(x) < 400:
            return np.zeros(AUDIO_DIM, np.float32), 0.0
        x = x[:sr * 12]
        rms = np.sqrt(np.mean(x*x) + 1e-10)
        frames = x[:(len(x)//400)*400].reshape(-1, 400)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1)+1e-10)
        zcr = np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))
        spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-8
        freqs = np.fft.rfftfreq(400, 1/sr)
        centroid = (spec * freqs).sum(axis=1) / spec.sum(axis=1)
        roll = (np.cumsum(spec, axis=1) < 0.85*spec.sum(axis=1, keepdims=True)).sum(axis=1) * sr / 400
        vals = [np.log1p(rms), np.log1p(np.std(x)), np.mean(np.abs(x)), zcr,
                np.log1p(np.mean(frame_rms)), np.std(frame_rms), np.max(frame_rms),
                np.mean(centroid)/8000, np.std(centroid)/8000, np.mean(roll)/8000,
                np.mean(frame_rms < .02), min(len(x)/sr, 12)/12]
        return np.nan_to_num(np.asarray(vals, np.float32)), 1.0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        return np.zeros(AUDIO_DIM, np.float32), 0.0


def vision_features(path):
    if not path or not Path(path).is_file():
        return np.zeros(VISION_DIM, np.float32), 0.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros(VISION_DIM, np.float32), 0.0
    samples = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return np.zeros(VISION_DIM, np.float32), 0.0
        for idx in np.linspace(0, total-1, min(6,total), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frame = cv2.resize(frame, (96, 96))
                samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255)
    finally:
        cap.release()
    if not samples:
        return np.zeros(VISION_DIM, np.float32), 0.0
    arr = np.stack(samples)
    motion = np.abs(np.diff(arr, axis=0)) if len(arr)>1 else np.zeros_like(arr[:1])
    center = arr[:, 24:72, 24:72]
    top = arr[:, :48]
    bottom = arr[:, 48:]
    vals = [arr.mean(), arr.std(), center.mean(), center.std(), top.mean(), bottom.mean(),
            np.mean(motion), np.std(motion), np.max(motion),
            np.mean(arr < .2), np.mean(arr > .8), min(len(samples),6)/6]
    return np.nan_to_num(np.asarray(vals, np.float32)), 1.0


def media_features(path):
    a, am = audio_features(path)
    v, vm = vision_features(path)
    return np.concatenate([a, v, [am, vm]]).astype(np.float32)
