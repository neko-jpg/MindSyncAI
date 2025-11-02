import os
import torch
import hydra
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, f1_score
from torchvision.ops import sigmoid_focal_loss

from data_loader import RavdessDataset
from features import LogMelSpectrogramExtractor
from model import CRNNModel

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, ema_model, data_loader, criterions, optimizer, device, cfg):
    """Runs a single training epoch and updates the EMA model."""
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Training", leave=False)
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)

        optimizer.zero_grad()
        outputs = model(features)

        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=cfg.dataset.num_classes).float()
        loss_emotion = sigmoid_focal_loss(outputs['emotion'], labels_one_hot, alpha=cfg.training.focal_loss.alpha, gamma=cfg.training.focal_loss.gamma, reduction='mean')
        loss_valence = criterions['valence'](outputs['valence'], valence)
        loss_arousal = criterions['arousal'](outputs['arousal'], arousal)

        weights = cfg.training.loss_weights
        loss = (weights.emotion * loss_emotion + weights.valence * loss_valence + weights.arousal * loss_arousal)

        loss.backward()
        optimizer.step()

        # Update EMA model after each step
        ema_model.update_parameters(model)

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model_to_eval, data_loader, criterions, device, cfg):
    """Evaluates a given model (can be the base model or EMA model)."""
    model_to_eval.eval()
    losses = {"total": 0, "emotion": 0, "valence": 0, "arousal": 0}
    all_preds, all_labels = [], []

    pbar = tqdm(data_loader, desc="Evaluating", leave=False)
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)

        outputs = model_to_eval(features)

        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=cfg.dataset.num_classes).float()
        loss_emotion = sigmoid_focal_loss(outputs['emotion'], labels_one_hot, alpha=cfg.training.focal_loss.alpha, gamma=cfg.training.focal_loss.gamma, reduction='mean')
        loss_valence = criterions['valence'](outputs['valence'], valence)
        loss_arousal = criterions['arousal'](outputs['arousal'], arousal)

        weights = cfg.training.loss_weights
        loss = (weights.emotion * loss_emotion + weights.valence * loss_valence + weights.arousal * loss_arousal)

        losses["total"] += loss.item()
        losses["emotion"] += loss_emotion.item()
        losses["valence"] += loss_valence.item()
        losses["arousal"] += loss_arousal.item()

        preds = torch.argmax(outputs['emotion'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    for key in losses: losses[key] /= len(data_loader)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return losses, uar, macro_f1

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---\n" + OmegaConf.to_yaml(cfg) + "---------------------")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    feature_extractor = LogMelSpectrogramExtractor(cfg)
    full_dataset = RavdessDataset(root_dir=data_path, feature_extractor=feature_extractor, cfg=cfg)

    speaker_ids = np.array([s['speaker_id'] for s in full_dataset.samples])
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.evaluation.test_size, random_state=cfg.seed)
    train_indices, val_indices = next(gss.split(full_dataset.samples, groups=speaker_ids))
    train_dataset, val_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    model = CRNNModel(cfg).to(device)
    # Initialize EMA model
    ema_model = AveragedModel(model)

    criterions = { 'valence': torch.nn.MSELoss(), 'arousal': torch.nn.MSELoss() }
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    log_history = []
    best_uar = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, ema_model, train_loader, criterions, optimizer, device, cfg)

        # Evaluate the base model for most of the training
        val_losses, uar, macro_f1 = evaluate(model, val_loader, criterions, device, cfg)
        scheduler.step()

        epoch_log = { "epoch": epoch, "train_loss": train_loss, **{f"val_{k}_loss": v for k, v in val_losses.items()}, "uar": uar, "macro_f1": macro_f1 }
        log_history.append(epoch_log)

        print(
            f"Epoch {epoch:02d}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_losses['total']:.4f} | "
            f"UAR: {uar:.4f} | Macro F1: {macro_f1:.4f}"
        )

        # Save the best *base* model during training
        if uar > best_uar:
            best_uar = uar
            # We save the EMA model's weights, as it's expected to generalize better
            torch.save(ema_model.module.state_dict(), "best_model.pth")
            print(f"  -> New best EMA model saved with UAR: {uar:.4f}")

    # --- Final Evaluation of the EMA model ---
    print("\n--- Final Evaluation of EMA Model ---")
    # Manually update bn statistics for the EMA model, compatible with dict-based dataloader
    print("Updating BatchNorm statistics for the EMA model...")
    ema_model.train()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Updating BN", leave=False):
            features = batch['features'].to(device)
            ema_model(features)

    val_losses_ema, uar_ema, macro_f1_ema = evaluate(ema_model, val_loader, criterions, device, cfg)
    print(
        f"Final EMA Model Performance | Val Loss: {val_losses_ema['total']:.4f} | "
        f"UAR: {uar_ema:.4f} | Macro F1: {macro_f1_ema:.4f}"
    )
    # Also log this final performance
    final_log = {"epoch": "final_ema", "train_loss": -1, **{f"val_{k}_loss": v for k, v in val_losses_ema.items()}, "uar": uar_ema, "macro_f1": macro_f1_ema}
    log_history.append(final_log)

    pd.DataFrame(log_history).to_csv("training_log.csv", index=False)
    print(f"\n--- Training Complete ---\nOutput directory: {os.getcwd()}")

if __name__ == "__main__":
    main()
