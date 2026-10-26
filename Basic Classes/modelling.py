from dataclasses import dataclass
from typing import Sequence
import numpy as np
from data_layer import SpeechEEGDataset
from features import AudioFeatureExtractor, EEGFeatureExtractor, Synchronizer


@dataclass
class FeatureTable:
    eeg: np.ndarray                 # n_eeg_features
    audio: np.ndarray               # n_audio_features
    y: np.ndarray                   # n_windows
    groups: np.ndarray              # participant IDs
    def __len__(self) -> int: ...
    def subset(self, idx: Sequence[int]) -> "FeatureTable": ...


def build_feature_table(dataset: SpeechEEGDataset, sync: Synchronizer,
                        eeg_extractor: EEGFeatureExtractor,
                        audio_extractor: AudioFeatureExtractor) -> FeatureTable: ...


class Classifier:
    ESTIMATORS: dict[str, object]
    kind: str
    model: object                   # sklearn Pipeline
    def __init__(self, kind: str = "logreg") -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...


class FeatureFusion:
    MODES: tuple[str, ...]          # "eeg", "audio", "early", "late"
    mode: str
    weights: np.ndarray
    clf: Classifier                 # used by "eeg", "audio", "early"
    eeg_clf: Classifier             # used by "late"
    audio_clf: Classifier           # used by "late"
    def __init__(self, mode: str = "early", kind: str = "logreg",
                 weights: tuple[float, float] = (0.5, 0.5)) -> None: ...
    def _X(self, t: FeatureTable) -> np.ndarray: ...
    def fit(self, t: FeatureTable) -> "FeatureFusion": ...
    def predict_proba(self, t: FeatureTable) -> np.ndarray: ...
    def predict(self, t: FeatureTable, threshold: float = 0.5) -> np.ndarray: ...
