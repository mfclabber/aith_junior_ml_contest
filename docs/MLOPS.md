# MLOps-стек

Пайплайн использует стандартные для индустрии инструменты: конфигурация,
трекинг экспериментов, версионирование данных, контроль качества кода.

## 1. Конфигурация — Hydra

Все параметры обучения вынесены в `conf/` и композируются из групп
(`data`, `model`, `train`, `tracking`). Переопределение из CLI без правки кода:

```bash
# продакшен (ViT-L-14, vlm6, TensorBoard)
python scripts/train.py

# лёгкий прогон на CPU для отладки
python scripts/train.py model=vit_b32 data=demo device=cpu tracking=none

# дообучение с warm-start + все бэкенды трекинга
python scripts/train.py train=finetune tracking=all train.epochs=200

# multirun (свип) по критерию отбора
python scripts/train.py -m train.select_by=acc,macro_f1,balanced_f1
```

Дерево конфигов:

```
conf/
├── config.yaml           # корень + defaults
├── data/{vlm6,demo}.yaml
├── model/{vit_l14,vit_b32}.yaml
├── train/{probe,finetune}.yaml
└── tracking/{tensorboard,all,none}.yaml
```

## 2. Трекинг экспериментов — TensorBoard / W&B / MLflow

Единый интерфейс `scripts/training/tracking.py` (`ExperimentTracker`)
пишет метрики сразу в выбранные бэкенды. Все бэкенды опциональны:
если пакет не установлен или выключен в конфиге — вызовы становятся no-op
(поэтому CI работает без тяжёлых зависимостей).

Логируются: `train_loss`, `val_acc`, `val_macro_f1`, `val_balanced_f1`
по эпохам для каждого weight-decay, сводка по сетке и итоговые метрики +
гиперпараметры и артефакт-чекпойнт.

```bash
make board       # TensorBoard  → http://localhost:6006
make mlflow-ui   # MLflow UI    → http://localhost:5000
# W&B — онлайн-дашборд (нужен wandb login)
```

Выбор бэкендов напрямую в bash-пайплайне:

```bash
TRACKING_BACKENDS=tensorboard,mlflow bash scripts/pipeline/05_train_probe.sh
```

## 3. Версионирование данных — DVC

`dvc.yaml` описывает стадии пайплайна декларативно (deps → outs).
DVC пересобирает только изменённое и хранит граф зависимостей:

```bash
dvc dag          # граф стадий
dvc repro        # пересобрать изменённые стадии
dvc metrics show # метрики из results/metrics.json
```

## 4. Качество кода — pre-commit, ruff, black

```bash
make install-dev     # ставит hydra, tensorboard, wandb, mlflow, dvc, pre-commit
pre-commit install   # хуки на git commit
make precommit       # прогнать по всем файлам
```

Конфигурация линтеров — в `pyproject.toml` (`[tool.ruff]`, `[tool.black]`),
хуки — в `.pre-commit-config.yaml`.

## Установка

```bash
make install-dev
```
