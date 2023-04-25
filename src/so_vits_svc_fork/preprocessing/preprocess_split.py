from __future__ import annotations

import math
from logging import getLogger
from pathlib import Path

import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

LOG = getLogger(__name__)


def _process_one(
    input_path: Path,
    output_dir: Path,
    sr: int,
    *,
    top_db: int = 30,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.1,
):
    try:
        audio, sr = librosa.load(input_path, sr=sr, mono=True)
    except Exception as e:
        LOG.warning(f"Failed to read {input_path}: {e}")
        return
    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=int(sr * frame_seconds),
        hop_length=int(sr * hop_seconds),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dur_t_sum = 0.0
    for start, end in tqdm(intervals, desc=f"Writing {input_path}"):
        audio_cut = audio[start:end]
        start_t = float(start) / sr
        end_t = float(end) / sr
        dur_t = end_t - start_t
        if 5.0 <= dur_t and dur_t <= 15.0:
            sf.write(
                (output_dir / f"{input_path.stem}_{start_t:.3f}_{end_t:.3f}.wav"),
                audio_cut,
                sr,
            )
            dur_t_sum += dur_t
    return dur_t_sum


def preprocess_split(
    input_dir: Path | str,
    output_dir: Path | str,
    sr: int,
    *,
    top_db: int = 30,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.1,
    n_jobs: int = -1,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(input_dir.rglob("*.*"))
    dur_t_sums = []
    with tqdm_joblib(desc="Splitting", total=len(input_paths)):
        dur_t_sums = Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(
                input_path,
                output_dir / input_path.relative_to(input_dir).parent,
                sr,
                top_db=top_db,
                frame_seconds=frame_seconds,
                hop_seconds=hop_seconds,
            )
            for input_path in input_paths
        )
    LOG.info(f"total valid duration (in seconds): {math.fsum(dur_t_sums):.3f}")
