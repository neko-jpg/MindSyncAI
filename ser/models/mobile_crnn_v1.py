import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileCRNNv1(nn.Module):
    """
    A lightweight, mobile-first CRNN architecture for Speech Emotion Recognition.
    This model is designed to handle variable-length audio inputs.

    Architecture:
    - 2 Convolutional layers for feature extraction
    - 1 Bidirectional GRU layer for temporal modeling
    - Global mean+max pooling for summarizing features
    - Fully connected layer for classification
    """
    def __init__(self, num_classes):
        super().__init__()

        # --- Convolutional Block ---
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # --- Recurrent Block ---
        # After two MaxPool2d layers, the feature dimension will be n_mels / 4
        # Assuming n_mels = 64, the input feature size will be 64 * (64/4) = 1024
        # We need to calculate this dynamically based on a dummy input.
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 64, 100) # (B, C, F, T)
            dummy_output = self.conv_block(dummy_input)
            rnn_input_dim = dummy_output.shape[1] * dummy_output.shape[2]

        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Classifier Head ---
        # The input to the classifier is the concatenation of mean and max pooling
        classifier_input_dim = (96 * 2) * 2 # (hidden_size * 2 for bidirectional) * 2 for (mean + max)
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x):
        # Input shape: (batch, n_mels, time)

        # Add channel dimension
        x = x.unsqueeze(1) # (batch, 1, n_mels, time)

        # Pass through convolutional block
        x = self.conv_block(x) # (batch, 64, n_mels/4, time/4)

        # Prepare for GRU: reshape and permute
        x = x.permute(0, 3, 1, 2) # (batch, time/4, 64, n_mels/4)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (batch, time/4, 64 * n_mels/4)

        # Flatten GRU parameters before forward pass.
        # This is a workaround for a known issue with torch.export and RNNs.
        self.gru.flatten_parameters()

        # Pass through GRU
        x, _ = self.gru(x) # (batch, time/4, 96 * 2)

        # Global Pooling (mean + max)
        mean_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)

        # Concatenate mean and max pooling
        pooled_output = torch.cat((mean_pool, max_pool), dim=1)

        # Classification
        logits = self.classifier(pooled_output)

        return logits

if __name__ == '__main__':
    import sys
    import os
    # Add project root to path to allow importing 'ser' modules
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from ser.preproc import load_spec

    # Load preproc spec to get n_mels
    spec = load_spec()
    n_mels = spec['n_mels']

    # Test the model with a dummy input
    num_classes = 8
    model = MobileCRNNv1(num_classes=num_classes)

    # Test with a variable-length input
    dummy_input_short = torch.randn(4, n_mels, 151) # ~1.5s
    dummy_input_long = torch.randn(4, n_mels, 300)  # ~3s

    output_short = model(dummy_input_short)
    output_long = model(dummy_input_long)

    print("✅ MobileCRNNv1 model test passed.")
    print(f"Input (short) shape: {dummy_input_short.shape} -> Output shape: {output_short.shape}")
    print(f"Input (long) shape: {dummy_input_long.shape} -> Output shape: {output_long.shape}")

    assert output_short.shape == (4, num_classes)
    assert output_long.shape == (4, num_classes)
    print("Output shapes are correct for variable-length inputs.")
