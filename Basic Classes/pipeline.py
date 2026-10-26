from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
from data_layer import SpeechEEGDataset
from evaluation import Evaluator
from features import AudioFeatureExtractor, EEGFeatureExtractor, Synchronizer
from modelling import FeatureFusion, FeatureTable


class Pipeline:
    dataset: SpeechEEGDataset
    sync: Synchronizer
    eeg_extractor: EEGFeatureExtractor
    audio_extractor: AudioFeatureExtractor
    evaluator: Evaluator
    cache_path: Optional[Path]
    def __init__(self, dataset: SpeechEEGDataset, sync: Optional[Synchronizer] = None,
                 eeg_extractor: Optional[EEGFeatureExtractor] = None,
                 audio_extractor: Optional[AudioFeatureExtractor] = None,
                 evaluator: Optional[Evaluator] = None,
                 cache_path: Optional[str | Path] = None) -> None: ...
    def build_table(self) -> FeatureTable: ...
    def compare(self, modes: Sequence[str] = FeatureFusion.MODES, kind: str = "logreg") -> pd.DataFrame: ...


def main() -> None: ...


