import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from data_layer import Participant, Recording
from features import AudioFeatureExtractor, EEGFeatureExtractor, EEGPreprocessor, Synchronizer
import mne



class EEGFeaturesTest(unittest.TestCase):
    def test_a_10hz_oscillation_shows_up_as_alpha_power_on_every_channel(self):
        sfreq, n_ch, seconds = 250.0, 4, 30
        t = np.arange(int(sfreq * seconds)) / sfreq
        rng = np.random.default_rng(0)
        data = 20e-6 * np.sin(2 * np.pi * 10 * t)[None, :] + 2e-6 * rng.normal(size=(n_ch, t.size))
        raw = mne.io.RawArray(data, mne.create_info([f"EEG{i}" for i in range(n_ch)], sfreq, "eeg"), verbose="ERROR")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub-raw.fif"
            raw.save(path, verbose="ERROR")
            rec = Recording(Participant("p1", 1), Path(tmp) / "unused.wav", path)
            extractor = EEGFeatureExtractor(EEGPreprocessor(notch_hz=None, n_ica_components=None))
            feats = extractor.extract(rec, [(0, 5), (5, 10)])
        self.assertEqual(feats.shape, (2, n_ch * 6))
        per_channel = feats.reshape(2, n_ch, 6)            # delta, theta, alpha, beta, gamma, entropy
        alpha, others = per_channel[:, :, 2], per_channel[:, :, [0, 1, 3, 4]]
        self.assertTrue((alpha[..., None] > others).all())


class AudioFeaturesTest(unittest.TestCase):
    def test_silence_raises_the_pause_ratio(self):
        import soundfile as sf
        sr = 16000
        t = np.arange(sr * 10) / sr
        tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32")
        half_silent = tone.copy()
        half_silent[: sr * 5] = 0
        with tempfile.TemporaryDirectory() as tmp:
            recs = {}
            for name, wave in (("tone", tone), ("half", half_silent)):
                sf.write(Path(tmp) / f"{name}.wav", wave, sr)
                recs[name] = Recording(Participant(name, 0), Path(tmp) / f"{name}.wav", Path(tmp) / "unused.fif")
            extractor = AudioFeatureExtractor(Synchronizer(5, 5))
            tone_f = extractor.extract(recs["tone"], [(0, 5), (5, 10)])
            half_f = extractor.extract(recs["half"], [(0, 5), (5, 10)])
        self.assertLess(tone_f[0, -1], 0.1)
        self.assertGreater(half_f[0, -1], 0.9)
        self.assertLess(half_f[1, -1], 0.1)


if __name__ == "__main__":
    unittest.main()
