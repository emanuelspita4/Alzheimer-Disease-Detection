from abc import ABC, abstractmethod
from typing import Optional, Sequence
import numpy as np
from data_layer import Recording

Window = tuple[float, float]        # (start_s, end_s) in EEG time
BANDS: dict[str, tuple[float, float]]


class Synchronizer:
    window_s: float
    step_s: float
    audio_offset_s: float
    def __init__(self, window_s: float = 5.0, step_s: float = 5.0, audio_offset_s: float = 0.0) -> None: ...
    def windows(self, eeg_duration: float, audio_duration: float) -> list[Window]: ...
    def for_recording(self, rec: Recording) -> list[Window]: ...
    def audio_window(self, w: Window) -> Window: ...


class EEGPreprocessor:
    l_freq: float
    h_freq: float
    notch_hz: Optional[float]
    n_ica_components: Optional[int]
    reference: str
    random_state: int
    def __init__(self, l_freq: float = 1.0, h_freq: float = 40.0, notch_hz: Optional[float] = 50.0,
                 n_ica_components: Optional[int] = 15, reference: str = "average",
                 random_state: int = 97) -> None: ...
    def process(self, raw): ...     # MNE raw -> cleaned MNE raw


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, rec: Recording, windows: Sequence[Window]) -> np.ndarray: ...   # (n_windows, n_features)


class EEGFeatureExtractor(FeatureExtractor):
    preprocessor: EEGPreprocessor
    def __init__(self, preprocessor: Optional[EEGPreprocessor] = None) -> None: ...
    def extract(self, rec: Recording, windows: Sequence[Window]) -> np.ndarray: ...


class AudioFeatureExtractor(FeatureExtractor):
    sync: Synchronizer              # maps EEG windows to audio time
    sr: int
    n_mfcc: int
    pause_top_db: float
    def __init__(self, synchronizer: Synchronizer, sr: int = 16000, n_mfcc: int = 13,
                 pause_top_db: float = 30) -> None: ...
    def extract(self, rec: Recording, windows: Sequence[Window]) -> np.ndarray: ...
