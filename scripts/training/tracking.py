"""Единый трекер экспериментов: TensorBoard + опц. Weights & Biases + MLflow.

Все бэкенды опциональны и подключаются мягко: если пакет не установлен или
бэкенд выключен, вызовы превращаются в no-op. Это позволяет запускать обучение
и в CI без тяжёлых зависимостей, и локально с полным трекингом.

    from scripts.training.tracking import ExperimentTracker
    tr = ExperimentTracker(backends=["tensorboard"], run_name="probe_vlm6")
    tr.log_params({"model": "ViT-L-14", "lr": 1e-2})
    tr.log_metrics({"val_acc": 0.76}, step=10)
    tr.log_summary({"best_macro_f1": 0.61})
    tr.close()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

log = logging.getLogger(__name__)

VALID_BACKENDS = ("tensorboard", "wandb", "mlflow")


class ExperimentTracker:
    """Тонкая обёртка над несколькими бэкендами трекинга."""

    def __init__(
        self,
        backends: Iterable[str] | None = None,
        run_name: str = "run",
        log_dir: str | Path = "runs",
        project: str = "jmlc-utt-classifier",
        config: Mapping | None = None,
    ) -> None:
        self.run_name = run_name
        self.project = project
        self._tb = None
        self._wandb = None
        self._mlflow = None
        self.backends: list[str] = []

        requested = [b.lower() for b in (backends or []) if b and b != "none"]
        for b in requested:
            if b not in VALID_BACKENDS:
                log.warning("unknown tracking backend: %s", b)
                continue
            init = getattr(self, f"_init_{b}")
            if init(Path(log_dir), config or {}):
                self.backends.append(b)

        if self.backends:
            log.info("tracking backends: %s", ", ".join(self.backends))
        else:
            log.info("tracking disabled (no active backends)")

    # ── init бэкендов (graceful) ──────────────────────────────────────
    def _init_tensorboard(self, log_dir: Path, config: Mapping) -> bool:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # noqa: BLE001
            log.warning("tensorboard unavailable: %s", exc)
            return False
        run_dir = log_dir / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._tb = SummaryWriter(log_dir=str(run_dir))
        return True

    def _init_wandb(self, log_dir: Path, config: Mapping) -> bool:
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb unavailable: %s", exc)
            return False
        try:
            self._wandb = wandb
            wandb.init(project=self.project, name=self.run_name,
                       config=dict(config), dir=str(log_dir), reinit=True)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb init failed: %s", exc)
            self._wandb = None
            return False

    def _init_mlflow(self, log_dir: Path, config: Mapping) -> bool:
        try:
            import mlflow
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow unavailable: %s", exc)
            return False
        try:
            self._mlflow = mlflow
            mlflow.set_tracking_uri(f"file:{Path(log_dir) / 'mlruns'}")
            mlflow.set_experiment(self.project)
            mlflow.start_run(run_name=self.run_name)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow init failed: %s", exc)
            self._mlflow = None
            return False

    # ── логирование ───────────────────────────────────────────────────
    def log_params(self, params: Mapping) -> None:
        clean = {k: v for k, v in params.items() if v is not None}
        if self._wandb is not None:
            self._wandb.config.update(clean, allow_val_change=True)
        if self._mlflow is not None:
            self._mlflow.log_params({k: str(v) for k, v in clean.items()})
        if self._tb is not None:
            text = "\n".join(f"{k}: {v}" for k, v in clean.items())
            self._tb.add_text("params", text)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if self._tb is not None:
            for k, v in clean.items():
                self._tb.add_scalar(k, v, global_step=step)
        if self._wandb is not None:
            self._wandb.log(clean, step=step)
        if self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step)

    def log_summary(self, summary: Mapping[str, float]) -> None:
        clean = {k: float(v) for k, v in summary.items() if v is not None}
        if self._tb is not None:
            for k, v in clean.items():
                self._tb.add_scalar(f"summary/{k}", v)
        if self._wandb is not None:
            self._wandb.summary.update(clean)
        if self._mlflow is not None:
            self._mlflow.log_metrics(clean)

    def log_artifact(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        if self._wandb is not None:
            self._wandb.save(str(p))
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(p))

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
