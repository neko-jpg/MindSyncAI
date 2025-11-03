import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T


def _time_shift_inplace(tensor: torch.Tensor, shift: int) -> None:
    """
    Circularly shift with zero padding so we do not wrap audio around.
    """
    if shift == 0:
        return
    if shift > 0:
        tensor[..., shift:] = tensor[..., :-shift]
        tensor[..., :shift] = 0
    else:
        shift = -shift
        tensor[..., :-shift] = tensor[..., shift:]
        tensor[..., -shift:] = 0


class FeatureAugmentor(nn.Module):
    """
    SpecAugment + ガウシアンノイズ + 時間シフトをログメル特徴に適用。
    """

    def __init__(
        self,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        noise_std: float = 0.0,
        time_shift_pct: float = 0.0,
    ):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param) if freq_mask_param > 0 else None
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param) if time_mask_param > 0 else None
        self.noise_std = noise_std
        self.time_shift_pct = time_shift_pct

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        if self.freq_mask is not None or self.time_mask is not None:
            x4d = x.unsqueeze(1)
            if self.freq_mask is not None:
                x4d = self.freq_mask(x4d)
            if self.time_mask is not None:
                x4d = self.time_mask(x4d)
            x = x4d.squeeze(1)

        if self.time_shift_pct > 0:
            max_shift = max(1, int(x.size(-1) * self.time_shift_pct))
            shifts = torch.randint(-max_shift, max_shift + 1, (x.size(0),), device=x.device)
            for i, shift in enumerate(shifts.tolist()):
                if shift > 0:
                    x[i, :, shift:] = x[i, :, :-shift]
                    x[i, :, :shift] = 0
                elif shift < 0:
                    shift = -shift
                    x[i, :, :-shift] = x[i, :, shift:]
                    x[i, :, -shift:] = 0

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x


class BatchAugmentor:
    """
    MixUp / SpecMix をバッチ単位で適用し、ラベルの線形補間情報を返す。
    """

    def __init__(
        self,
        mixup_alpha: float,
        mixup_prob: float,
        specmix_prob: float,
        specmix_segments: int,
        specmix_alpha: float,
    ):
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.specmix_prob = specmix_prob
        self.specmix_segments = specmix_segments
        self.specmix_alpha = specmix_alpha

    def _mixup(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        perm = torch.randperm(features.size(0), device=features.device)
        mixed_features = lam * features + (1 - lam) * features[perm]
        return mixed_features, labels, labels[perm], lam

    def _specmix(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        b, f, t = features.shape
        perm = torch.randperm(b, device=features.device)
        mixed = features.clone()
        total_mask_area = 0

        for _ in range(self.specmix_segments):
            freq_size = max(1, int(torch.rand(1).item() * f * self.specmix_alpha))
            time_size = max(1, int(torch.rand(1).item() * t * self.specmix_alpha))

            freq_start = random.randint(0, max(0, f - freq_size))
            time_start = random.randint(0, max(0, t - time_size))

            freq_end = freq_start + freq_size
            time_end = time_start + time_size

            mixed[:, freq_start:freq_end, time_start:time_end] = features[perm, freq_start:freq_end, time_start:time_end]
            total_mask_area += freq_size * time_size

        lam = max(0.0, 1.0 - min(1.0, total_mask_area / (f * t)))
        return mixed, labels, labels[perm], lam

    def __call__(self, features: torch.Tensor, labels: torch.Tensor):
        if features.size(0) < 2:
            return features, labels, None

        if self.mixup_alpha > 0 and torch.rand(1).item() < self.mixup_prob:
            mixed_features, y_a, y_b, lam = self._mixup(features, labels)
            return mixed_features, labels, {"y_a": y_a, "y_b": y_b, "lam": lam}

        if self.specmix_prob > 0 and torch.rand(1).item() < self.specmix_prob:
            mixed_features, y_a, y_b, lam = self._specmix(features, labels)
            return mixed_features, labels, {"y_a": y_a, "y_b": y_b, "lam": lam}

        return features, labels, None


class TTAAugmentor:
    """
    テスト時の平均化予測。特徴量と必要に応じて波形にノイズ・時間シフトを与える。
    """

    def __init__(self, enabled: bool, samples: int, noise_std: float, time_shift_pct: float):
        self.enabled = enabled
        self.samples = samples
        self.noise_std = noise_std
        self.time_shift_pct = time_shift_pct

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        aug = features.clone()
        if self.noise_std > 0:
            aug = aug + torch.randn_like(aug) * self.noise_std
        if self.time_shift_pct > 0:
            max_shift = max(1, int(aug.size(-1) * self.time_shift_pct))
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                aug[:, :, shift:] = aug[:, :, :-shift]
                aug[:, :, :shift] = 0
            elif shift < 0:
                shift = -shift
                aug[:, :, :-shift] = aug[:, :, shift:]
                aug[:, :, -shift:] = 0
        return aug

    def _augment_waveforms(self, waveforms: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if waveforms is None:
            return None
        aug = waveforms.clone()
        if self.noise_std > 0:
            aug = aug + torch.randn_like(aug) * self.noise_std
        if self.time_shift_pct > 0:
            max_shift = max(1, int(aug.size(-1) * self.time_shift_pct))
            shift = random.randint(-max_shift, max_shift)
            _time_shift_inplace(aug, shift)
        return aug

    def __call__(
        self,
        model: nn.Module,
        features: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled or self.samples <= 0:
            return model(features=features, waveforms=waveforms, waveform_lengths=waveform_lengths)

        outputs = []
        for _ in range(self.samples):
            aug_features = self._augment_features(features)
            aug_waveforms = self._augment_waveforms(waveforms)
            outputs.append(
                model(
                    features=aug_features,
                    waveforms=aug_waveforms,
                    waveform_lengths=waveform_lengths,
                )
            )

        return torch.stack(outputs, dim=0).mean(dim=0)
