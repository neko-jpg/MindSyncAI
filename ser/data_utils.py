from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from data import CombinedSERDataset, CremaDDataset, RavdessDataset


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


def _create_dataset_instance(
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

        dataset = _create_dataset_instance(
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
