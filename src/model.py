import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class AttentionPooling(nn.Module):
    """Attention pooling layer."""
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        scores = self.v(u).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context

class CRNNModel(nn.Module):
    """
    A CRNN for multi-task SER, with SpecAugment integrated for robust training.
    """
    def __init__(self, cfg):
        super().__init__()

        # --- SpecAugment Block (applied only during training) ---
        self.spec_augment = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=cfg.training.spec_augment.freq_mask_param),
            T.TimeMasking(time_mask_param=cfg.training.spec_augment.time_mask_param),
        )

        # --- Convolutional Block ---
        conv_layers = []
        in_channels = 1
        for out_channels in cfg.model.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        self.conv_block = nn.Sequential(*conv_layers)

        # --- Recurrent Block ---
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, cfg.features.n_mels, 151)
            dummy_output = self.conv_block(dummy_input)
            rnn_input_dim = dummy_output.shape[1] * dummy_output.shape[2]

        self.gru = nn.GRU(
            rnn_input_dim, cfg.model.gru_hidden_size,
            num_layers=2, batch_first=True, bidirectional=True
        )

        # --- Multi-Head Prediction Block ---
        shared_feature_dim = cfg.model.gru_hidden_size * 2
        self.attention = AttentionPooling(shared_feature_dim, cfg.model.attention_dim)
        self.emotion_classifier = nn.Linear(shared_feature_dim, cfg.dataset.num_classes)
        self.valence_regressor = nn.Linear(shared_feature_dim, 1)
        self.arousal_regressor = nn.Linear(shared_feature_dim, 1)

    def forward(self, x):
        # x shape: (batch, n_mels, time)

        # Apply SpecAugment only during training
        if self.training:
            x = self.spec_augment(x)

        x = x.unsqueeze(1) # Add channel dim
        x = self.conv_block(x)

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x, _ = self.gru(x)
        shared_features = self.attention(x)

        emotion_logits = self.emotion_classifier(shared_features)
        valence = torch.tanh(self.valence_regressor(shared_features).squeeze(-1))
        arousal = torch.tanh(self.arousal_regressor(shared_features).squeeze(-1))

        return {
            'emotion': emotion_logits,
            'valence': valence,
            'arousal': arousal
        }

if __name__ == '__main__':
    from omegaconf import OmegaConf

    dummy_cfg = OmegaConf.create({
        'features': {'n_mels': 64},
        'model': {
            'conv_channels': [32, 64, 64],
            'gru_hidden_size': 128,
            'attention_dim': 128
        },
        'dataset': {'num_classes': 8},
        'training': {
            'spec_augment': { 'freq_mask_param': 24, 'time_mask_param': 50 }
        }
    })

    model = CRNNModel(cfg=dummy_cfg)
    dummy_input = torch.randn(4, 64, 151)

    # Test in training mode (SpecAugment should be applied)
    model.train()
    output_train = model(dummy_input)
    print("Test in train mode passed.")

    # Test in evaluation mode (SpecAugment should be skipped)
    model.eval()
    output_eval = model(dummy_input)
    print("Test in eval mode passed.")

    assert output_train['emotion'].shape == output_eval['emotion'].shape
    print("\nAll output shapes are correct.")
