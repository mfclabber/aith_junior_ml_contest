.PHONY: help install install-dev demo test lint precommit pipeline pipeline-quick \
        train train-cfg train-sweep board mlflow-ui eval serve docker agent clean

PY ?= ./venv/bin/python
PIP ?= ./venv/bin/pip
RUNS_DIR ?= runs

help:
	@echo "JMLC — классификация участков по панорамам"
	@echo ""
	@echo "  make install        — venv + базовые зависимости (web+ml)"
	@echo "  make install-dev    — + dev/mlops/agent (pytest, ruff, hydra, tensorboard, ...)"
	@echo "  make test           — unit-тесты (CI)"
	@echo "  make lint           — syntax + ruff"
	@echo "  make precommit      — прогнать pre-commit по всем файлам"
	@echo "  make demo           — UI в режиме heuristic (без GPU)"
	@echo "  make pipeline-quick — EDA → train probe на demo-данных (CPU)"
	@echo "  make pipeline       — полный продакшен-пайплайн (нужны data/)"
	@echo "  make train          — обучение probe vlm6 (bash-скрипт)"
	@echo "  make train-cfg      — обучение через Hydra (conf/) + трекинг"
	@echo "  make train-sweep    — Hydra multirun по select_by (пример свипа)"
	@echo "  make board          — TensorBoard на $(RUNS_DIR)"
	@echo "  make mlflow-ui      — MLflow UI на runs/mlruns"
	@echo "  make eval           — бенчмарк чекпойнтов"
	@echo "  make serve          — веб-пилот ML"
	@echo "  make agent          — LangGraph metric agent"
	@echo "  make docker         — docker compose up"

install:
	python3 -m venv venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

test:
	$(PY) -m pytest tests/ -v --tb=short

lint:
	$(PY) -m compileall -q web_app scripts tests
	@command -v ruff >/dev/null && ruff check web_app scripts tests || echo "ruff not installed (make install-dev)"

precommit:
	@command -v pre-commit >/dev/null && pre-commit run --all-files || echo "pre-commit not installed (make install-dev)"

demo:
	CLASSIFIER_MODE=heuristic CLASSIFY_REQUIRE_PANORAMA=0 $(PY) -m web_app.app

pipeline-quick:
	bash scripts/pipeline/run_quick.sh

pipeline:
	bash scripts/pipeline/run_production.sh

train:
	bash scripts/pipeline/05_train_probe.sh

train-cfg:
	$(PY) scripts/train.py

train-sweep:
	$(PY) scripts/train.py -m train.select_by=acc,macro_f1,balanced_f1

board:
	$(PY) -m tensorboard.main --logdir $(RUNS_DIR)

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri file:$(RUNS_DIR)/mlruns

eval:
	bash scripts/pipeline/06_evaluate.sh

serve:
	bash scripts/run_web_classifier.sh

agent:
	bash scripts/agent/run_metric_agent.sh

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov outputs multirun
