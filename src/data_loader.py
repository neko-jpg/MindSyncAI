import os
from glob import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from features import LogMelSpectrogramExtractor

class RavdessDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS. Handles audio preprocessing and returns data
    for multi-task learning (classification + regression).
    """

    def __init__(self, root_dir, feature_extractor, cfg):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.target_sr = cfg.features.sample_rate
        self.target_len_samples = int(self.target_sr * cfg.features.segment_duration_s)
        self.resamplers = {}

        # --- Valence-Arousal Mapping based on Russell's Circumplex Model ---
        # Values are normalized between -1 and 1.
        self.va_map = {
            0: {'name': 'neutral',   'valence': 0.0, 'arousal': 0.0},
            1: {'name': 'calm',      'valence': 0.7, 'arousal': -0.7},
            2: {'name': 'happy',     'valence': 0.7, 'arousal': 0.7},
            3: {'name': 'sad',       'valence': -0.7,'arousal': -0.7},
            4: {'name': 'angry',     'valence': -0.7,'arousal': 0.7},
            5: {'name': 'fearful',   'valence': -0.6,'arousal': 0.8},
            6: {'name': 'disgust',   'valence': -0.8,'arousal': 0.6},
            7: {'name': 'surprised', 'valence': 0.0, 'arousal': 1.0},
        }
        # Emotion label to class index mapping
        self.emotion_map = {
            "01": 0, "02": 1, "03": 2, "04": 3,
            "05": 4, "06": 5, "07": 6, "08": 7
        }
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        """Creates a list of sample metadata."""
        samples = []
        filepaths = glob(os.path.join(self.root_dir, 'Actor_*', '*.wav'))
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            parts = filename.split('.')[0].split('-')
            emotion_idx = self.emotion_map[parts[2]]
            samples.append({
                'filepath': filepath,
                'label': emotion_idx,
                'speaker_id': int(parts[6]) - 1,
                'valence': self.va_map[emotion_idx]['valence'],
                'arousal': self.va_map[emotion_idx]['arousal']
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_resampler(self, input_sr):
        if input_sr not in self.resamplers:
            self.resamplers[input_sr] = torchaudio.transforms.Resample(
                orig_freq=input_sr, new_freq=self.target_sr
            )
        return self.resamplers[input_sr]

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        waveform, sample_rate = torchaudio.load(sample_info['filepath'])

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.target_sr:
            waveform = self._get_resampler(sample_rate)(waveform)

        current_len = waveform.shape[1]
        if current_len > self.target_len_samples:
            start = torch.randint(0, current_len - self.target_len_samples + 1, (1,)).item()
            waveform = waveform[:, start:start + self.target_len_samples]
        elif current_len < self.target_len_samples:
            padding = self.target_len_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        features = self.feature_extractor(waveform).squeeze(0)

        return {
            'features': features,
            'label': torch.tensor(sample_info['label'], dtype=torch.long),
            'valence': torch.tensor(sample_info['valence'], dtype=torch.float),
            'arousal': torch.tensor(sample_info['arousal'], dtype=torch.float),
            'speaker_id': sample_info['speaker_id']
        }

if __name__ == '__main__':
    # Test for the multi-task data loader
    dataset_path = os.path.join('data', 'RAVDESS')
    if os.path.exists(dataset_path):
        dummy_cfg = OmegaConf.create({
            'features': {
                'sample_rate': 16000,
                'segment_duration_s': 1.5,
                'win_length_ms': 25, 'hop_length_ms': 10, 'n_mels': 64
            }
        })

        extractor = LogMelSpectrogramExtractor(cfg=dummy_cfg)
        dataset = RavdessDataset(root_dir=dataset_path, feature_extractor=extractor, cfg=dummy_cfg)

        print(f"Successfully initialized dataset with {len(dataset)} samples.")
        sample = dataset[0]

        print("\nSample 0 content:")
        for key, value in sample.items():
            print(f"  - {key}: {value.shape if isinstance(value, torch.Tensor) else value}")

        assert 'valence' in sample and 'arousal' in sample
        print("\nTest passed: Valence and Arousal values are included.")
    else:
        print(f"Dataset directory not found: {dataset_path}")
