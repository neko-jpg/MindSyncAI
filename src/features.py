import torch
import torch.nn as nn
import torchaudio.transforms as T

class LogMelSpectrogramExtractor(nn.Module):
    """
    A feature extractor that converts raw audio waveforms into log-mel spectrograms.
    It assumes the input waveform is already at the target sample rate.
    """
    def __init__(self, cfg):
        """
        Initializes the transformation pipeline using parameters from the config.

        Args:
            cfg (DictConfig): A Hydra configuration object with a 'features' section.
        """
        super().__init__()
        self.cfg = cfg
        self.target_sr = cfg.features.sample_rate

        win_length = int(self.target_sr * cfg.features.win_length_ms / 1000)
        hop_length = int(self.target_sr * cfg.features.hop_length_ms / 1000)

        self.mel_spectrogram = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=win_length,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=cfg.features.n_mels
            ),
            T.AmplitudeToDB(stype='power', top_db=80)
        )

    def forward(self, waveform):
        """
        Applies the feature extraction pipeline.

        Args:
            waveform (Tensor): An input audio waveform at the target sample rate.

        Returns:
            Tensor: The resulting log-mel spectrogram.
        """
        return self.mel_spectrogram(waveform)

if __name__ == '__main__':
    from omegaconf import OmegaConf

    dummy_cfg_dict = {
        'features': {
            'sample_rate': 16000,
            'win_length_ms': 25,
            'hop_length_ms': 10,
            'n_mels': 64
        }
    }
    dummy_cfg = OmegaConf.create(dummy_cfg_dict)
    extractor = LogMelSpectrogramExtractor(cfg=dummy_cfg)

    # Create a dummy audio tensor already at the target sample rate
    input_sr = 16000
    dummy_waveform = torch.randn(1, input_sr * 3)

    log_mel_spec = extractor(dummy_waveform)

    print("Feature extractor test passed.")
    print(f"Input shape: {dummy_waveform.shape} -> Output shape: {log_mel_spec.shape}")
