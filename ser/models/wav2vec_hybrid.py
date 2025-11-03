from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModel
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("transformers ライブラリが必要です。`pip install transformers` を実行してください。") from exc


class Wav2Vec2SERNet(nn.Module):
    """
    Wav2Vec2 などの事前学習音声モデルを取り込み、自己注意 + MLP ヘッドで分類するモデル。
    """

    def __init__(self, cfg):
        super().__init__()
        pretrained_cfg = cfg.pretrained
        model_name = pretrained_cfg.model_name
        cache_dir = pretrained_cfg.cache_dir if pretrained_cfg.cache_dir not in (None, "") else None

        transformer_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=pretrained_cfg.local_files_only,
            trust_remote_code=pretrained_cfg.trust_remote_code,
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=pretrained_cfg.local_files_only,
            trust_remote_code=pretrained_cfg.trust_remote_code,
            config=transformer_config,
        )

        if pretrained_cfg.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = transformer_config.hidden_size
        dropout = pretrained_cfg.dropout
        self.freeze_encoder = pretrained_cfg.freeze_encoder
        self.unfreeze_at_epoch = pretrained_cfg.unfreeze_at_epoch
        self.encoder_requires_grad = not self.freeze_encoder
        if self.freeze_encoder:
            self._set_encoder_trainable(False)

        self.attn = nn.MultiheadAttention(hidden_size, pretrained_cfg.attention_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

        head_input_dim = hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, pretrained_cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pretrained_cfg.head_hidden_dim, cfg.dataset.num_classes),
        )

    def _set_encoder_trainable(self, flag: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = flag
        self.encoder_requires_grad = flag

    def update_freezing(self, epoch: int) -> None:
        if self.freeze_encoder and not self.encoder_requires_grad and epoch >= self.unfreeze_at_epoch:
            self._set_encoder_trainable(True)

    def forward(
        self,
        features: Optional[torch.Tensor] = None,
        waveforms: Optional[torch.Tensor] = None,
        waveform_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if waveforms is None:
            raise ValueError("Wav2Vec2SERNet では waveforms テンソルが必須です。")

        if waveform_lengths is not None:
            max_len = waveforms.size(1)
            attention_mask = (
                torch.arange(max_len, device=waveforms.device)
                .unsqueeze(0)
                .expand(waveforms.size(0), -1)
                < waveform_lengths.unsqueeze(1)
            ).long()
        else:
            attention_mask = None

        encoder_outputs = self.encoder(waveforms, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        attn_out, _ = self.attn(hidden_states, hidden_states, hidden_states, need_weights=False)
        hidden_states = hidden_states + attn_out
        hidden_states = self.ff(hidden_states)

        mean_pool = hidden_states.mean(dim=1)
        max_pool, _ = hidden_states.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        logits = self.head(pooled)
        return logits
