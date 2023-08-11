from __future__ import annotations

import math
from logging import getLogger
from pathlib import Path

import librosa
import numpy
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

LOG = getLogger(__name__)

class AudioSegment:
    def __init__(self,
                 start: numpy.int64 = None,
                 end: numpy.int64 = None,
                 sample_rate: int = None,
    ):
        if start is None or end is None or sample_rate is None:
            self._segments = []
            self.sample_rate = 0
            self.duration = float(0)
        else:
            self._segments = [(start, end)]
            self.sample_rate = sample_rate
            self.duration = float(end - start) / sample_rate

    def __add__(self, other):
        new_seg = AudioSegment()
        new_seg._segments = self._segments + other._segments
        new_seg.sample_rate = max([self.sample_rate, other.sample_rate])
        new_seg.duration = self.duration + other.duration
        return new_seg

    def save(self, audio: numpy.ndarray, output_dir: Path, file_prefix: str):
        if len(self._segments) == 0:
            LOG.warning("Empty segments are ignored to save")
            return
        seg = numpy.concatenate([audio[start:end] for start, end in self._segments])
        file_path = output_dir / f"{file_prefix}_{self.start():.3f}_{self.duration:.3f}.wav"
        sf.write(file_path, seg, self.sample_rate)

    def start(self):
        if len(self._segments) == 0:
            return float(-1)
        return float(self._segments[0][0]) / self.sample_rate

def _process_one_ori(
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


def _process_one(
    input_path: Path,
    output_dir: Path,
    sr: int,
    *,
    max_length: float = 10.0,
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
    composed_audio_seg = AudioSegment()
    for start, end in tqdm(intervals, desc=f"Writing {input_path}"):
        audio_seg = AudioSegment(start, end, sr)
        new_composed_audio_seg = composed_audio_seg + audio_seg
        # the 5s - 15s slice is suggested by:
        # https://github.com/svc-develop-team/so-vits-svc#0-slice-audio
        if new_composed_audio_seg.duration <= 15.0:
            composed_audio_seg = new_composed_audio_seg
        else:
            if 5.0 <= composed_audio_seg.duration and composed_audio_seg.duration <= 15.0:
                composed_audio_seg.save(audio, output_dir, input_path.stem)
                dur_t_sum += composed_audio_seg.duration
            else:
                LOG.info('drop seg,'
                         f' start {composed_audio_seg.start()},'
                         f' duration {composed_audio_seg.duration}')
            composed_audio_seg = audio_seg
    return dur_t_sum


def preprocess_split(
    input_dir: Path | str,
    output_dir: Path | str,
    sr: int,
    *,
    max_length: float = 10.0,
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
                max_length=max_length,
                top_db=top_db,
                frame_seconds=frame_seconds,
                hop_seconds=hop_seconds,
            )
            for input_path in input_paths
        )
    LOG.info(f"total valid duration (in seconds): {math.fsum(dur_t_sums):.3f}")
