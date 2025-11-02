# Speech Emotion Recognition with PyTorch

This repository provides a complete pipeline for training and exporting a Speech Emotion Recognition (SER) model. The entire workflow, from feature extraction to model training and deployment, is implemented in PyTorch. The project is designed to be extensible, allowing for the easy integration of new datasets and model architectures.

## Key Features

- **End-to-End PyTorch Workflow**: The entire pipeline, including feature extraction, training, and inference, is built with PyTorch and `torchaudio`.
- **Unified Preprocessing**: A consistent and unified feature extraction process (`log-mel spectrogram`) is used for both training and inference, eliminating inconsistencies.
- **Extensible Data Loaders**: The data loading structure is designed to be modular, allowing for the easy addition of new datasets like RAVDESS and CREMA-D.
- **Mobile-First Model**: Includes a lightweight, mobile-first CRNN model (`MobileCRNNv1`) designed for efficient on-device inference.
- **PyTorch Mobile Export**: Provides a script to export the trained model to the PyTorch Mobile (TorchScript) format for deployment on iOS and Android.

## Project Structure

```
.
├── export/                  # Scripts for exporting the model
│   └── export_torchscript.py
├── ser/                     # Main source code for the SER project
│   ├── conf/                # Hydra configuration files
│   ├── data/                # Dataset loaders
│   ├── models/              # Model architectures
│   ├── __init__.py
│   ├── main.py              # Main training script
│   └── preproc.py           # Unified preprocessing module
├── test/                    # Tests for the project
├── models/                  # Saved models
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 1. Setup

### a. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### b. Install Dependencies
It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.
```bash
pip install -r requirements.txt
```

### c. Prepare Datasets
1. Download the [RAVDESS](https://zenodo.org/record/1188976) and/or [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) datasets.
2. Place the datasets in a `data/` directory at the project's root. The expected structure is:
   ```
   .
   ├── data/
   │   ├── RAVDESS/
   │   │   ├── Actor_01/
   │   │   └── ...
   │   └── CREMA-D/
   │       ├── 1001_DFA_ANG_XX.wav
   │       └── ...
   └── ...
   ```

## 2. Training the Model

The training process is managed by [Hydra](https://hydra.cc/). You can customize the training parameters by editing the configuration file at `ser/conf/config.yaml` or by overriding them from the command line.

To start training, run the main script from the project root:
```bash
python -m ser.main
```

To override parameters, use the following syntax:
```bash
python -m ser.main training.epochs=50 dataset.name=ravdess
```

The training script will save the best model (based on Unweighted Average Recall - UAR) to `best_model.pth` and log the training progress to `training_log.csv`.

## 3. Exporting the Model for Mobile

To deploy the model on a mobile device, you need to convert it to the PyTorch Mobile (TorchScript) format.

Run the `export_torchscript.py` script, pointing it to the checkpoint of your trained model:
```bash
python export/export_torchscript.py --checkpoint_path best_model.pth --output_path models/mobile_crnn_v1.ptl
```

The final, optimized model will be saved at `models/mobile_crnn_v1.ptl`, ready to be integrated into an iOS or Android application using the PyTorch Mobile SDK.
