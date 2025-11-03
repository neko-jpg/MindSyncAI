import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, ListConfig
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data_utils import build_dataset
from data import RavdessDataset, CremaDDataset, CombinedSERDataset
from .losses import build_loss
from .models.hybrid_ser import HybridSERNet
from .models.mobile_crnn_v1 import MobileCRNNv1
from .models.wav2vec_hybrid import Wav2Vec2SERNet
from .training.augmentations import BatchAugmentor, FeatureAugmentor, TTAAugmentor
from .training.callbacks import EarlyStopping
from .training.distillation import DistillationHelper
from .training.ema import build_ema

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback
    StratifiedGroupKFold = None
from sklearn.model_selection import StratifiedKFold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_batch(batch):
    features = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    speaker_ids = torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long)

    feature_lengths = torch.tensor([feat.shape[-1] for feat in features], dtype=torch.long)
    max_feat_len = int(feature_lengths.max())
    padded_features = []
    for feat in features:
        pad_amount = max_feat_len - feat.shape[-1]
        if pad_amount > 0:
            feat = F.pad(feat, (0, pad_amount))
        padded_features.append(feat)
    feature_tensor = torch.stack(padded_features)

    if "waveform" in batch[0]:
        waveforms = [item["waveform"] for item in batch]
        waveform_lengths = torch.tensor([item["waveform_length"] for item in batch], dtype=torch.long)
        max_wave_len = int(waveform_lengths.max())
        waveform_tensor = torch.zeros(len(batch), max_wave_len, dtype=waveforms[0].dtype)
        for idx, waveform in enumerate(waveforms):
            waveform_tensor[idx, : waveform.shape[-1]] = waveform
    else:
        waveform_tensor = None
        waveform_lengths = None

    return {
        "features": feature_tensor,
        "label": labels,
        "speaker_id": speaker_ids,
        "length": feature_lengths,
        "waveform": waveform_tensor,
        "waveform_length": waveform_lengths,
    }


def compute_class_counts(dataset, indices: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for idx in indices:
        label = dataset.samples[idx]["label"]
        counts[label] += 1
    return counts


def _ensure_list_config(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_source_cfg(dataset_cfg: DictConfig, name: str):
    sources = dataset_cfg.get("sources")
    if sources is not None and name in sources:
        return sources[name]
    return dataset_cfg.get(name)


def _dataset_root_from_cfg(default_root: str, source_dict: Dict[str, object]) -> str:
    candidate = source_dict.get("root") or source_dict.get("path") or default_root
    return hydra.utils.to_absolute_path(str(candidate))


def create_dataset_instance(
    name: str,
    dataset_root: str,
    cfg: DictConfig,
    segment_duration: Optional[float],
    hop_duration: Optional[float],
    min_coverage: float,
):
    name = name.lower()
    if name == "ravdess":
        return RavdessDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    if name in {"cremad", "crema_d"}:
        return CremaDDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    raise ValueError(f"Unsupported dataset name: {name}")


def build_dataset(cfg: DictConfig):
    dataset_cfg = cfg.dataset
    dataset_names = _ensure_list_config(dataset_cfg.get("names"))
    if not dataset_names:
        dataset_names = _ensure_list_config(dataset_cfg.get("name"))
    if not dataset_names:
        raise ValueError("dataset.name or dataset.names must be specified in the configuration.")

    segment_default = dataset_cfg.get("segment_duration_s", None)
    hop_default = dataset_cfg.get("hop_duration_s", None)
    min_cov_default = dataset_cfg.get("min_segment_coverage", 0.6)
    default_root = dataset_cfg.get("root") or dataset_cfg.get("path") or cfg.data_dir

    datasets = []
    for raw_name in dataset_names:
        name = str(raw_name).lower()
        source_cfg = _resolve_source_cfg(dataset_cfg, name) or {}
        if isinstance(source_cfg, DictConfig):
            source_dict = OmegaConf.to_container(source_cfg, resolve=True)
        elif isinstance(source_cfg, dict):
            source_dict = dict(source_cfg)
        else:
            source_dict = {}

        dataset_root = _dataset_root_from_cfg(default_root, source_dict)
        segment_duration = source_dict.get("segment_duration_s", segment_default)
        hop_duration = source_dict.get("hop_duration_s", hop_default)
        min_coverage = source_dict.get("min_segment_coverage", min_cov_default)

        dataset = create_dataset_instance(
            name=name,
            dataset_root=dataset_root,
            cfg=cfg,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage if min_coverage is not None else min_cov_default,
        )
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    return CombinedSERDataset(datasets)


def build_model(cfg, device: torch.device):
    name = cfg.model.name.lower()
    if name == "hybrid_ser":
        model = HybridSERNet(cfg)
    elif name == "mobile_crnn":
        model = MobileCRNNv1(num_classes=cfg.dataset.num_classes)
    elif name == "wav2vec2_hybrid":
        model = Wav2Vec2SERNet(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    return model.to(device)


def create_optimizer(cfg, model):
    optimizer_cfg = cfg.training.optimizer
    name = optimizer_cfg.name.lower()
    if name == "adamw":
        return AdamW(
            model.parameters(),
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay,
            betas=tuple(optimizer_cfg.betas),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")


def create_scheduler(cfg, optimizer):
    sched_cfg = cfg.training.scheduler
    warmup_epochs = sched_cfg.warmup_epochs
    schedulers = []
    milestones = []
    if warmup_epochs and warmup_epochs > 0:
        schedulers.append(LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=max(1, sched_cfg.t_max - warmup_epochs),
            eta_min=sched_cfg.eta_min,
        )
    )
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def refresh_batch_norm(model, data_loader: DataLoader, device: torch.device) -> None:
    model.train()
    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            waveforms = batch.get("waveform")
            if waveforms is not None:
                waveforms = waveforms.to(device)
                waveform_lengths = batch["waveform_length"].to(device)
            else:
                waveform_lengths = None
            lengths = batch["length"].to(device)
            model(
                features=features,
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
                lengths=lengths,
            )
    model.eval()


def evaluate(
    model,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
    tta: Optional[TTAAugmentor] = None,
    temperature: Optional[torch.Tensor] = None,
    collect_logits: bool = False,
) -> Dict[str, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    logits_list = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            targets = batch["label"].to(device)
            lengths = batch["length"].to(device)
            waveforms = batch.get("waveform")
            waveform_lengths = batch.get("waveform_length")
            if waveforms is not None:
                waveforms = waveforms.to(device)
            if waveform_lengths is not None:
                waveform_lengths = waveform_lengths.to(device)

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

            if temperature is not None:
                outputs = outputs / temperature

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=-1)
            pred = torch.argmax(probabilities, dim=-1)
            preds.append(pred.cpu().numpy())
            labels.append(targets.cpu().numpy())

            if collect_logits:
                logits_list.append(outputs.cpu())

    preds = np.concatenate(preds)
    labels_np = np.concatenate(labels)
    avg_loss = total_loss / max(1, len(data_loader))
    uar = recall_score(labels_np, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

    result = {"loss": avg_loss, "uar": uar, "macro_f1": macro_f1}
    if collect_logits and logits_list:
        result["logits"] = torch.cat(logits_list)
        result["labels"] = torch.from_numpy(labels_np)
    return result


def tune_temperature(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    device = logits.device
    log_temp = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(min=1e-3)
        loss = F.cross_entropy(logits / temperature, labels.to(device))
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temp.detach().exp().clamp(min=1e-3)


def maybe_build_distiller(cfg, device: torch.device):
    distill_cfg = cfg.training.distillation
    if not distill_cfg.enabled or not distill_cfg.teacher_checkpoint:
        return None

    teacher = build_model(cfg, device)
    ckpt_path = hydra.utils.to_absolute_path(distill_cfg.teacher_checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        teacher.load_state_dict(state["state_dict"])
    else:
        teacher.load_state_dict(state)
    teacher.eval()
    print(f"[Distillation] Teacher loaded from {ckpt_path}")
    return DistillationHelper(
        teacher_model=teacher,
        temperature=distill_cfg.temperature,
        alpha=distill_cfg.alpha,
    )


def build_batch_augmentor(cfg) -> Optional[BatchAugmentor]:
    mix_cfg = cfg.training.mixup
    specmix_cfg = cfg.training.specmix
    if mix_cfg.alpha <= 0 and specmix_cfg.prob <= 0:
        return None
    return BatchAugmentor(
        mixup_alpha=mix_cfg.alpha,
        mixup_prob=mix_cfg.prob,
        specmix_prob=specmix_cfg.prob,
        specmix_segments=specmix_cfg.segments,
        specmix_alpha=specmix_cfg.alpha,
    )


def train_single_fold(
    cfg,
    dataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    device: torch.device,
    fold_name: str,
):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    class_counts = compute_class_counts(dataset, train_indices, cfg.dataset.num_classes)
    model = build_model(cfg, device)
    criterion = build_loss(cfg, class_counts, device)
    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    feature_augmentor = FeatureAugmentor(
        freq_mask_param=cfg.training.spec_augment.freq_mask_param,
        time_mask_param=cfg.training.spec_augment.time_mask_param,
        noise_std=cfg.training.feature_aug.noise_std,
        time_shift_pct=cfg.training.feature_aug.time_shift_pct,
    )
    batch_augmentor = build_batch_augmentor(cfg)
    ema_model = build_ema(model, cfg.training.ema.decay).to(device) if cfg.training.ema.enabled else None
    distiller = maybe_build_distiller(cfg, device)
    early_stopper = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        mode="max",
    )
    tta = TTAAugmentor(
        enabled=cfg.evaluation.tta.enabled,
        samples=cfg.evaluation.tta.samples,
        noise_std=cfg.evaluation.tta.noise_std,
        time_shift_pct=cfg.evaluation.tta.time_shift_pct,
    ) if cfg.evaluation.tta.enabled else None

    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", fold_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    meta_path = os.path.join(ckpt_dir, "best_model_meta.json")
    log_path = os.path.join(ckpt_dir, "training_log.csv")

    log_history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    best_source = "base"

    print(f"\n--- Training {fold_name} ---")
    for epoch in range(1, cfg.training.epochs + 1):
        if hasattr(model, "update_freezing"):
            model.update_freezing(epoch)

        model.train()
        total_loss = 0.0
        pbar = train_loader

        for i, batch in enumerate(pbar):
            print(f"[{fold_name}] Epoch {epoch:03d} | Batch {i+1}/{len(pbar)}", end='\r')
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["length"].to(device)
            waveforms = batch.get("waveform")
            waveform_lengths = batch.get("waveform_length")
            if waveforms is not None:
                waveforms = waveforms.to(device)
            if waveform_lengths is not None:
                waveform_lengths = waveform_lengths.to(device)

            features = feature_augmentor(features)
            mix_info = None
            if batch_augmentor is not None:
                features, labels, mix_info = batch_augmentor(features, labels)

            optimizer.zero_grad()
            outputs = model(
                features=features,
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
                lengths=lengths,
            )

            if mix_info is not None:
                loss = mix_info["lam"] * criterion(outputs, mix_info["y_a"]) + (1 - mix_info["lam"]) * criterion(
                    outputs, mix_info["y_b"]
                )
            else:
                loss = criterion(outputs, labels)

            if distiller is not None:
                kd_loss = distiller(
                    outputs,
                    features,
                    waveforms=waveforms,
                    waveform_lengths=waveform_lengths,
                )
                loss = (1.0 - distiller.alpha) * loss + distiller.alpha * kd_loss

            loss.backward()
            if cfg.training.gradient_clip:
                clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()

            if ema_model is not None:
                ema_model.update_parameters(model)

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader))

        base_metrics = evaluate(model, val_loader, criterion, device)
        candidate_metrics = base_metrics
        candidate_state = model.state_dict()
        candidate_source = "base"

        if ema_model is not None:
            refresh_batch_norm(ema_model, train_loader, device)
            ema_metrics = evaluate(ema_model, val_loader, criterion, device)
            if ema_metrics["uar"] >= base_metrics["uar"]:
                candidate_metrics = ema_metrics
                candidate_state = ema_model.state_dict()
                candidate_source = "ema"

        log_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": candidate_metrics["loss"],
                "val_uar": candidate_metrics["uar"],
                "val_macro_f1": candidate_metrics["macro_f1"],
                "source": candidate_source,
            }
        )

        print(
            f"[{fold_name}] Epoch {epoch:03d} | Train {train_loss:.4f} | "
            f"Val UAR {candidate_metrics['uar']:.4f} | Val F1 {candidate_metrics['macro_f1']:.4f} | Source {candidate_source}"
        )

        if candidate_metrics["uar"] > best_metric:
            best_metric = candidate_metrics["uar"]
            best_source = candidate_source
            state_to_save = {k: v.detach().cpu() for k, v in candidate_state.items()}
            torch.save(state_to_save, best_model_path)
            print(f"  -> New best ({best_source.upper()}) model saved to {best_model_path}")

        if early_stopper.step(candidate_metrics["uar"]):
            print(f"[{fold_name}] Early stopping triggered.")
            break

    pd.DataFrame(log_history).to_csv(log_path, index=False)

    best_model = build_model(cfg, device)
    best_state = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_state)

    final_metrics = evaluate(
        best_model,
        val_loader,
        criterion,
        device,
        tta=None,
        collect_logits=cfg.evaluation.calibration.enabled,
    )

    if cfg.evaluation.calibration.enabled:
        logits = final_metrics.pop("logits").to(device)
        labels_tensor = final_metrics.pop("labels").to(device)
        temperature = tune_temperature(logits, labels_tensor)
        temperature_value = temperature.item()
        print(f"[{fold_name}] Temperature calibrated: {temperature_value:.4f}")
    else:
        temperature = torch.tensor(1.0, device=device)
        temperature_value = 1.0

    tta_metrics = evaluate(
        best_model,
        val_loader,
        criterion,
        device,
        tta=tta,
        temperature=temperature,
    )

    meta = {
        "fold": fold_name,
        "best_source": best_source,
        "temperature": temperature_value,
        "val_loss": tta_metrics["loss"],
        "val_uar": tta_metrics["uar"],
        "val_macro_f1": tta_metrics["macro_f1"],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{fold_name}] Final | Loss {tta_metrics['loss']:.4f} | UAR {tta_metrics['uar']:.4f} | "
        f"Macro F1 {tta_metrics['macro_f1']:.4f}"
    )

    return {
        "metrics": tta_metrics,
        "meta_path": meta_path,
        "checkpoint": best_model_path,
        "log_path": log_path,
    }


def build_cv_splits(cfg, dataset) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = np.array([sample["label"] for sample in dataset.samples])
    speakers = np.array([sample["speaker_id"] for sample in dataset.samples])

    if cfg.evaluation.cv.enabled:
        n_splits = cfg.evaluation.cv.folds
        if StratifiedGroupKFold is not None and cfg.evaluation.cv.stratified:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=cfg.evaluation.cv.shuffle,
                random_state=cfg.seed,
            )
            splits = splitter.split(labels, labels, groups=speakers)
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=cfg.evaluation.cv.shuffle,
                random_state=cfg.seed,
            )
            splits = splitter.split(labels, labels)
        return [(np.array(train_idx), np.array(val_idx)) for train_idx, val_idx in splits]

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.evaluation.test_size, random_state=cfg.seed)
    train_indices, val_indices = next(gss.split(dataset.samples, groups=speakers))
    return [(np.array(train_indices), np.array(val_indices))]


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---\n" + OmegaConf.to_yaml(cfg) + "---------------------")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(cfg)

    splits = build_cv_splits(cfg, dataset)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_name = f"fold_{fold_idx}" if cfg.evaluation.cv.enabled else "holdout"
        result = train_single_fold(cfg, dataset, train_idx, val_idx, device, fold_name)
        fold_results.append(result)

    if len(fold_results) > 1:
        uars = [res["metrics"]["uar"] for res in fold_results]
        f1s = [res["metrics"]["macro_f1"] for res in fold_results]
        print("\n=== Cross-Validation Summary ===")
        print(f"Mean UAR : {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"Mean F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print("\n--- Training Complete ---")
    for res in fold_results:
        print(f"Checkpoint: {res['checkpoint']}")
        print(f"Metadata  : {res['meta_path']}")
        print(f"Log       : {res['log_path']}")


if __name__ == "__main__":
    main()
