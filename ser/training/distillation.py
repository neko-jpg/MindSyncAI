import torch
import torch.nn.functional as F


class DistillationHelper:
    """
    教師モデルのロジットをソフトターゲットとして利用するためのヘルパー。
    """

    def __init__(self, teacher_model, temperature: float, alpha: float):
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def __call__(self, student_logits: torch.Tensor, features: torch.Tensor, waveforms=None, waveform_lengths=None):
        with torch.no_grad():
            teacher_logits = self.teacher(
                features=features,
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
            )
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        return kd_loss
