import csv
from pathlib import Path
import numpy as np
from modelling import FeatureTable


def write_dataset(root: Path, rows):
    root = Path(root)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "eeg").mkdir(parents=True, exist_ok=True)
    csv_path = root / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "label", "audio_path", "eeg_path", "age", "sex"])
        for pid, label, audio, eeg in rows:
            (root / "audio" / audio).touch()
            (root / "eeg" / eeg).touch()
            w.writerow([pid, label, f"audio/{audio}", f"eeg/{eeg}", 70, "F"])
    return csv_path


def make_table(n_participants=20, windows=6, separable=True, seed=0) -> FeatureTable:
    rng = np.random.default_rng(seed)
    y_participant = np.array([0, 1] * (n_participants // 2))
    rows = np.repeat(np.arange(n_participants), windows)
    y = y_participant[rows]
    shift = 1.5 if separable else 0.0
    eeg = rng.normal(size=(len(y), 6)) + shift * y[:, None]
    audio = rng.normal(size=(len(y), 4)) + 0.5 * shift * y[:, None]
    groups = np.array([f"p{i:02d}" for i in rows])
    return FeatureTable(eeg, audio, y, groups)
