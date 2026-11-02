from dataclasses import dataclass
from typing import Callable, Iterator
import numpy as np
import pandas as pd
from modelling import FeatureFusion, FeatureTable


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict: ...


@dataclass
class EvalResult:
    window: dict
    participant: dict
    predictions: pd.DataFrame       # one row per participant
    def summary(self) -> dict: ...


class Evaluator:
    n_splits: int
    loso: bool
    seed: int
    threshold: float
    def __init__(self, n_splits: int = 5, loso: bool = False, seed: int = 0,
                 threshold: float = 0.5) -> None: ...
    def _splits(self, table: FeatureTable) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
    def evaluate(self, table: FeatureTable,
                 make_model: Callable[[], FeatureFusion]) -> EvalResult: ...
