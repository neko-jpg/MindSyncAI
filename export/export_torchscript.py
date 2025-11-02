import torch
import argparse
import os
import sys

# Add project root to path to allow importing 'ser' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ser.models.mobile_crnn_v1 import MobileCRNNv1
from torch.utils.mobile_optimizer import optimize_for_mobile

def export_to_torchscript(model, dummy_input, output_path):
    """
    Exports a PyTorch model to TorchScript format for mobile deployment.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use torch.jit.trace to create a TorchScript model
    traced_model = torch.jit.trace(model, dummy_input)

    # Optimize the model for mobile
    optimized_model = optimize_for_mobile(traced_model)

    # Save the optimized model
    optimized_model._save_for_lite_interpreter(output_path)

    print(f"✅ Model successfully exported to TorchScript (PyTorch Mobile) format at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export a PyTorch SER model to TorchScript.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes for the model.")
    parser.add_argument("--output_path", type=str, default="models/mobile_crnn_v1.ptl", help="Path to save the TorchScript model.")

    args = parser.parse_args()

    # --- Model Loading ---
    model = MobileCRNNv1(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval() # Set the model to evaluation mode

    # --- Dummy Input ---
    # The dummy input is not used for scripting, but it's good practice to have it
    dummy_input = torch.randn(1, 64, 151, requires_grad=False)

    # --- Export ---
    export_to_torchscript(model, dummy_input, args.output_path)

if __name__ == '__main__':
    main()
