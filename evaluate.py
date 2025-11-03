import glob
import json
import os
from typing import List, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.ravdess_dataset import RavdessDataset
from ser.models.hybrid_ser import HybridSERNet
from ser.models.mobile_crnn_v1 import MobileCRNNv1
from ser.models.wav2vec_hybrid import Wav2Vec2SERNet
from ser.training.augmentations import TTAAugmentor


def collate_batch(batch):
    features = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    feature_lengths = torch.tensor([feat.shape[-1] for feat in features], dtype=torch.long)
    max_feat_len = int(feature_lengths.max())
    padded_features = []
    for feat in features:
        pad_amount = max_feat_len - feat.shape[-1]
        if pad_amount > 0:
            feat = F.pad(feat, (0, pad_amount))
        padded_features.append(feat)
    feature_tensor = torch.stack(padded_features)

    waveforms = [item["waveform"] for item in batch]
    waveform_lengths = torch.tensor([item["waveform_length"] for item in batch], dtype=torch.long)
    max_wave_len = int(waveform_lengths.max())
    waveform_tensor = torch.zeros(len(batch), max_wave_len, dtype=waveforms[0].dtype)
    for idx, waveform in enumerate(waveforms):
        waveform_tensor[idx, : waveform.shape[-1]] = waveform

    return {
        "features": feature_tensor,
        "label": labels,
        "length": feature_lengths,
        "waveform": waveform_tensor,
        "waveform_length": waveform_lengths,
    }


def build_model(cfg, device: torch.device):
    name = cfg.model.name.lower()
    if name == "hybrid_ser":
        model = HybridSERNet(cfg)
    elif name == "mobile_crnn":
        model = MobileCRNNv1(num_classes=cfg.dataset.num_classes)
    elif name == "wav2vec2_hybrid":
        model = Wav2Vec2SERNet(cfg)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    return model.to(device)


def ensure_checkpoints(cfg) -> List[str]:
    checkpoints = cfg.evaluation.ensemble.checkpoints
    if checkpoints:
        return [hydra.utils.to_absolute_path(path) for path in checkpoints]

    ckpt_root = hydra.utils.to_absolute_path("checkpoints")
    holdout_ckpt = os.path.join(ckpt_root, "best_model.pth")
    if os.path.exists(holdout_ckpt):
        return [holdout_ckpt]

    fold_ckpts = sorted(glob.glob(os.path.join(ckpt_root, "fold_*", "best_model.pth")))
    if not fold_ckpts:
        raise FileNotFoundError("No checkpoints found under 'checkpoints/' and none specified in the config.")
    return fold_ckpts


def load_temperature(ckpt_path: str) -> float:
    meta_path = os.path.join(os.path.dirname(ckpt_path), "best_model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            return float(meta.get("temperature", 1.0))
    return 1.0


def predict(
    model,
    data_loader: DataLoader,
    device: torch.device,
    temperature: float,
    tta: Optional[TTAAugmentor],
):
    model.eval()
    probs_list = []
    labels_list = []

    temperature_tensor = torch.tensor(temperature, device=device)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            features = batch["features"].to(device)
            labels = batch["label"].cpu().numpy()
            lengths = batch["length"].to(device)
            waveforms = batch["waveform"].to(device)
            waveform_lengths = batch["waveform_length"].to(device)

            outputs = (
                tta(model, features, waveforms, waveform_lengths)
                if tta is not None
                else model(
                    features=features,
                    waveforms=waveforms,
                    waveform_lengths=waveform_lengths,
                    lengths=lengths,
                )
            )
            outputs = outputs / temperature_tensor
            probs = torch.softmax(outputs, dim=-1)
            probs_list.append(probs.cpu())
            labels_list.append(labels)

    probs = torch.cat(probs_list)
    labels = np.concatenate(labels_list)
    return probs, labels


@hydra.main(config_path="ser/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Evaluation ---")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = hydra.utils.to_absolute_path(cfg.data_dir)
    dataset = RavdessDataset(
        data_dir=data_path,
        sample_rate=cfg.features.sample_rate,
        n_mels=cfg.features.n_mels,
    )

    speaker_ids = np.array([sample["speaker_id"] for sample in dataset.samples])
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.evaluation.test_size, random_state=cfg.seed)
    _, val_indices = next(splitter.split(dataset.samples, groups=speaker_ids))
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    tta = TTAAugmentor(
        enabled=cfg.evaluation.tta.enabled,
        samples=cfg.evaluation.tta.samples,
        noise_std=cfg.evaluation.tta.noise_std,
        time_shift_pct=cfg.evaluation.tta.time_shift_pct,
    ) if cfg.evaluation.tta.enabled else None

    checkpoints = ensure_checkpoints(cfg)
    print(f"Loaded {len(checkpoints)} checkpoint(s).")

    ensemble_probs = []
    temperature_values = []

    for ckpt_path in checkpoints:
        temperature = load_temperature(ckpt_path)
        temperature_values.append(temperature)

        model = build_model(cfg, device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

        probs, labels = predict(model, val_loader, device, temperature, tta)
        ensemble_probs.append(probs)

    avg_probs = torch.stack(ensemble_probs).mean(dim=0)
    preds = torch.argmax(avg_probs, dim=-1).numpy()

    accuracy = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    print("\n--- Metrics ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"UAR      : {uar:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    if temperature_values:
        print(f"Temperatures used: {[round(t, 4) for t in temperature_values]}")


if __name__ == "__main__":
    main()
