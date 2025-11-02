import torch
import torch.nn as nn
import torchaudio.transforms as T
import yaml
from pathlib import Path

def load_spec(spec_path="preproc_spec.yaml"):
    """Loads the preprocessing specification from a YAML file."""
    # Build the full path to the spec file relative to this script.
    base_dir = Path(__file__).parent
    full_path = base_dir / spec_path
    with open(full_path, 'r') as f:
        spec = yaml.safe_load(f)
    return spec

class LogMelSpectrogram(nn.Module):
    """
    Unified feature extractor that converts raw audio waveforms into log-mel spectrograms.
    This module is configured from a YAML specification file.
    """
    def __init__(self, spec=None):
        """
        Initializes the transformation pipeline using parameters from the spec.
        If no spec is provided, it loads the default 'preproc_spec.yaml'.
        """
        super().__init__()
        if spec is None:
            self.spec = load_spec()
        else:
            self.spec = spec

        # Convert window and hop lengths from ms to samples
        win_length = int(self.spec['sample_rate'] * self.spec['win_length_ms'] / 1000)
        hop_length = int(self.spec['sample_rate'] * self.spec['hop_length_ms'] / 1000)

        self.mel_spectrogram = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.spec['sample_rate'],
                n_fft=self.spec.get('n_fft', win_length), # Use n_fft from spec or default to win_length
                win_length=win_length,
                hop_length=hop_length,
                n_mels=self.spec['n_mels']
            ),
            T.AmplitudeToDB(stype='power', top_db=self.spec['top_db'])
        )

    def forward(self, waveform):
        """
        Applies the feature extraction pipeline.

        Args:
            waveform (Tensor): An input audio waveform.

        Returns:
            Tensor: The resulting log-mel spectrogram.
        """
        # Handle empty waveform input
        if waveform.shape[-1] == 0:
            batch_size = waveform.shape[0]
            n_mels = self.spec['n_mels']
            # Return an empty spectrogram with shape [batch, n_mels, 0]
            return torch.zeros((batch_size, n_mels, 0), dtype=waveform.dtype, device=waveform.device)

        return self.mel_spectrogram(waveform)

if __name__ == '__main__':
    # Test the feature extractor with a dummy audio tensor
    extractor = LogMelSpectrogram()

    sample_rate = extractor.spec['sample_rate']
    dummy_waveform = torch.randn(2, sample_rate * 3) # Batch of 2, 3 seconds long

    log_mel_spec = extractor(dummy_waveform)

    print("✅ LogMelSpectrogram module test passed.")
    print(f"Input shape: {dummy_waveform.shape}")
    print(f"Output shape: {log_mel_spec.shape}")

    # Expected time frames: (3 * 16000) / 160 (hop_length) = 300. It's often 301 due to padding.
    print(f"Expected time frames: ~{int((sample_rate * 3) / (sample_rate * extractor.spec['hop_length_ms'] / 1000)) + 1}")
    print(f"Expected mel bins: {extractor.spec['n_mels']}")
