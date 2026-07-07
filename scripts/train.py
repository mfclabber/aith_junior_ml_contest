#!/usr/bin/env python3
"""Hydra-точка входа для обучения CLIP linear probe.

Композиция конфига из conf/ и запуск обучения с трекингом экспериментов.

Примеры:
  python scripts/train.py
  python scripts/train.py model=vit_b32 data=demo device=cpu tracking=none
  python scripts/train.py train=finetune tracking=all train.epochs=200
  python scripts/train.py -m train.select_by=acc,macro_f1,balanced_f1   # multirun

Под капотом переиспользуется scripts/training/linear_probe_clip.py (argparse),
поэтому вся логика обучения/кэширования фич — в одном месте.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import linear_probe_clip as probe


def _build_argv(cfg: DictConfig) -> list[str]:
    argv = [
        "linear_probe_clip.py",
        "--metadata", str(cfg.data.metadata),
        "--data-dir", str(cfg.data.data_dir),
        "--model", str(cfg.model.name),
        "--pretrained", str(cfg.model.pretrained),
        "--group", str(cfg.data.group),
        "--bottom-crop", str(cfg.data.bottom_crop),
        "--val-cap", str(cfg.data.val_cap),
        "--select-by", str(cfg.train.select_by),
        "--epochs", str(cfg.train.epochs),
        "--finetune-lr", str(cfg.train.finetune_lr),
        "--device", str(cfg.device),
        "--runs-dir", str(cfg.paths.runs_dir),
    ]
    if cfg.train.get("save"):
        argv += ["--save", str(cfg.train.save)]
    if cfg.train.get("init_head"):
        argv += ["--init-head", str(cfg.train.init_head)]
    backends = list(cfg.tracking.get("backends", []) or [])
    if backends:
        argv += ["--tracking", ",".join(backends)]
        argv += ["--experiment", f"{cfg.data.name}_{cfg.model.name}_{cfg.train.mode}"]
    return argv


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    sys.argv = _build_argv(cfg)
    probe.main()


if __name__ == "__main__":
    main()
