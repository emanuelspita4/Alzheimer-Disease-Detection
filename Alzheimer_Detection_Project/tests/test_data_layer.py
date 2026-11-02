import tempfile
import unittest
from pathlib import Path
from data_layer import SpeechEEGDataset
from helpers import write_dataset


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_from_csv_builds_recordings_and_summary(self):
        csv = write_dataset(self.root, [("p1", 1, "a1.wav", "e1.fif"), ("p1", 1, "a2.wav", "e2.fif"),("p2", 0, "a3.wav", "e3.fif")])
        ds = SpeechEEGDataset.from_csv(csv)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds.summary(), {"recordings": 3, "participants": 2, "alzheimers": 1, "controls": 1})
        self.assertEqual(ds[0].participant.id, "p1")

    def test_same_participant_shares_one_participant_object(self):
        csv = write_dataset(self.root, [("p1", 1, "a1.wav", "e1.fif"), ("p1", 1, "a2.wav", "e2.fif")])
        ds = SpeechEEGDataset.from_csv(csv)
        self.assertIs(ds[0].participant, ds[1].participant)

    def test_conflicting_labels_for_one_participant_are_rejected(self):
        csv = write_dataset(self.root, [("p1", 1, "a1.wav", "e1.fif"), ("p1", 0, "a2.wav", "e2.fif")])
        with self.assertRaises(ValueError):
            SpeechEEGDataset.from_csv(csv)

    def test_missing_file_is_reported(self):
        csv = write_dataset(self.root, [("p1", 1, "a1.wav", "e1.fif")])
        (self.root / "audio" / "a1.wav").unlink()
        with self.assertRaises(FileNotFoundError):
            SpeechEEGDataset.from_csv(csv)
        SpeechEEGDataset.from_csv(csv, check_files=False)

    def test_missing_required_column_is_rejected(self):
        bad = self.root / "bad.csv"
        bad.write_text("participant_id,label\np1,1\n")
        with self.assertRaises(ValueError):
            SpeechEEGDataset.from_csv(bad)

    def _ten_participants(self):
        rows = [(f"p{i}", i % 2, f"a{i}_{k}.wav", f"e{i}_{k}.fif") for i in range(12) for k in range(2)]
        return SpeechEEGDataset.from_csv(write_dataset(self.root, rows))

    def test_grouped_kfold_never_splits_a_participant(self):
        ds = self._ten_participants()
        for train, test in ds.grouped_kfold(n_splits=4):
            self.assertTrue(set(ds.groups[train]).isdisjoint(ds.groups[test]))

    def test_leave_one_subject_out_has_one_participant_per_test_fold(self):
        ds = self._ten_participants()
        folds = list(ds.leave_one_subject_out())
        self.assertEqual(len(folds), 12)
        for _, test in folds:
            self.assertEqual(len(set(ds.groups[test])), 1)


if __name__ == "__main__":
    unittest.main()
