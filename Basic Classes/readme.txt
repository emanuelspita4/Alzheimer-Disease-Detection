Notes regarding the methodology and designs pattern used

The split in Python script is as follows:
    -data layer
        --Participant - for identifying a patient as well as additional personal information
        --Recording - links a participant to their audio and EEG files
        --SpeechEEGDataset - loads and indexes all recordings; also keeps participant-grouped train/test splits
    -features
        --EEGPreprocessor - takes a EEG object, returns clean mne.Epochs
        --EEGFeatureExtractor - power, entropy, connectivity, other from epochs into a feature vector
        --AudioFeatureExtractor - the same but for audio
        --Synchronizer - aligns audio timestamps with EEG events
    -modeling
        --FeatureFusion - computes probabilities
        --Classifier - a wrapper forscikit-learn - fit, predict, and predict_proba
    -evaluation
        --Evaluator - runs cross-validation
        --EvalResult - keeps result
    -pipeline
        --Pipeline - ties dataset, features, modeling, evaluation

Design patterns:
    -behavioral
        --EEGFeatureExtractor and AudioFeatureExtractor are interchangeable as implementations
        --SpeechEEGDatasetis iterable
    -structural
        --Classifier gives scikit-learn estimators one consistent interface
        --inheritance is used only for the extractors, everything else is composed, for example EEGFeatureExtractor holding an EEGPreprocessor
    -architectural
        --data flows through stages: dataset -> windows -> features -> fusion -> evaluation
        --Participant and Recording are frozen data classes, and FeatureTable and EvalResult are plain data carriers passed between stages
        --Recording reads files only when asked
