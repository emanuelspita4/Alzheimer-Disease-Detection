from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class Participant:
    id: str
    forename: str
    surname: str
    label: int                      # 0 - no Alzheimer detected, 1 - potential Alzheimer detected, 2 - near-certain Alzheimer detected; other types of grading might end up being used, potentially not integers
    age: Optional[float] = None


@dataclass(frozen=True)
class Recording:
    participant: Participant        # for one participant ther could be multiple recordings
    audio_path: Path
    eeg_path: Path
    def load_audio(self, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]: ...
    def load_eeg(self, preload: bool = True): ...  # returns mne.io.Raw


class SpeechEEGDataset:
    REQUIRED_COLUMNS: set[str]
    recordings: list[Recording]
    def __init__(self, recordings: Sequence[Recording]) -> None: ...

    @classmethod
    def from_csv(cls, csv_path: str | Path, root: str | Path | None = None,
                 check_files: bool = True) -> "SpeechEEGDataset": ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Recording: ...
    def __iter__(self) -> Iterator[Recording]: ...

    @property
    def labels(self) -> np.ndarray: ...

    @property
    def groups(self) -> np.ndarray: ...
    def subset(self, indices: Sequence[int]) -> "SpeechEEGDataset": ...
    def grouped_kfold(self, n_splits: int = 5, seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
    def leave_one_subject_out(self) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
    def summary(self) -> dict: ...
