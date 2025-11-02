import torch
import unittest
import sys
import os

# Add the 'ser' directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ser.preproc import LogMelSpectrogram

class TestPreprocessingConsistency(unittest.TestCase):
    """
    Tests the consistency and correctness of the preprocessing module.
    """
    def setUp(self):
        """Set up a feature extractor and a dummy waveform for testing."""
        self.extractor = LogMelSpectrogram()
        self.spec = self.extractor.spec
        self.sample_rate = self.spec['sample_rate']

        # Create a dummy waveform: batch size=2, 2.5 seconds of audio
        self.dummy_waveform = torch.randn(2, int(self.sample_rate * 2.5))

    def test_output_shape(self):
        """
        Tests if the output spectrogram has the correct shape.
        """
        log_mel_spec = self.extractor(self.dummy_waveform)

        # Expected shape: [batch_size, n_mels, time_frames]
        batch_size, n_mels, time_frames = log_mel_spec.shape

        self.assertEqual(batch_size, self.dummy_waveform.shape[0])
        self.assertEqual(n_mels, self.spec['n_mels'])

        # Calculate expected time frames
        hop_length = int(self.spec['sample_rate'] * self.spec['hop_length_ms'] / 1000)
        expected_time_frames = (self.dummy_waveform.shape[1] // hop_length) + 1

        self.assertEqual(time_frames, expected_time_frames)

    def test_reproducibility(self):
        """
        Tests if the feature extraction is deterministic (reproducible).
        """
        log_mel_spec_1 = self.extractor(self.dummy_waveform)
        log_mel_spec_2 = self.extractor(self.dummy_waveform)

        # The output should be exactly the same for the same input
        self.assertTrue(torch.equal(log_mel_spec_1, log_mel_spec_2))

    def test_empty_input(self):
        """
        Tests the behavior of the extractor with an empty input tensor.
        """
        empty_waveform = torch.randn(1, 0)
        log_mel_spec = self.extractor(empty_waveform)

        # Expect the time dimension to be 0 for an empty input
        self.assertEqual(log_mel_spec.shape[2], 0)

if __name__ == '__main__':
    unittest.main()
