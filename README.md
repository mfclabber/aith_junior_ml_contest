# Классификатор городских участков по уличным панорамам

[![CI](https://github.com/mfclabber/aith_junior_ml_contest/actions/workflows/ci.yml/badge.svg)](https://github.com/mfclabber/aith_junior_ml_contest/actions)

Проект для [Junior ML Contest 2026](https://ai.itmo.ru/junior_ml_contest) (AI Talent Hub, ITMO).
Автор — Дмитрий Новичков.

## О чём это

Аналитики девелопера вручную просматривают тысячи земельных участков в Москве и
относят каждый к одному из типов городской территории (UTT): активная застройка,
стройка, деградирующий или недоиспользуемый фонд, природные зоны. Работа медленная
и плохо масштабируется.

Спутник тут помогает слабо — он не видит дворы и фасады. Текстовое описание из
GPKG тоже не спасает: например, 1712 участков имеют одну и ту же фразу «зелень во
дворах», но по факту относятся к разным классам. Отличить их можно только по виду
с улицы.

Идея проекта простая: взять уличную панораму участка и по ней предсказать класс.
Вокруг этого собран полный конвейер — от сбора данных до веб-инструмента, в котором
аналитик проверяет и правит результат, а правки возвращаются в обучение.

Модель различает **6 классов UTT**:

| Класс | Что это |
|-------|---------|
| `natural_areas` | природные территории, зелёные зоны |
| `low_density_degraded` | малоэтажная / деградирующая застройка |
| `underused_infrastructure` | недоиспользуемая инфраструктура |
| `frozen_construction` | замороженная стройка |
| `active_construction` | активная стройка |
| `active_urban` | сформированная городская застройка |

## Как это работает

По координатам участка из GPKG система находит точки панорам (по URL или вдоль
граней полигона), вырезает перспективные кропы 896×672, прогоняет их через CLIP
ViT-L-14 и обученный поверх него линейный классификатор. Поскольку у одного участка
обычно несколько ракурсов, финальный класс выбирается голосованием с весами по
уверенности.

<!-- ```mermaid
flowchart LR
  A[OSM + панорамы] --> B[EDA и аудит]
  B --> C[Чистка + кропы]
  C --> D[VLM-разметка]
  D --> E[CLIP linear probe]
  E --> F[Оценка + веб-UI]
``` -->

Отдельно стоит отметить два решения, которые заметно повлияли на качество:

- **Мульти-азимут.** Панорама в точке участка часто смотрела «мимо» (bearing = 0),
  таких мусорных кропов было около 29%. Стали рендерить четыре направления
  (0/90/180/270) и выбирать релевантные.
- **VLM вместо сырых OSM-меток.** Согласие меток OSM с реальным содержимым кадра —
  всего ~20%, поэтому метки переразмечены VLM-судьёй (Qwen) в режиме weak
  supervision, а спорные случаи вычитывались вручную.

Подробнее — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/PIPELINE.md`](docs/PIPELINE.md) и [`docs/EDA.md`](docs/EDA.md).

## Быстрый старт

```bash
git clone https://github.com/mfclabber/aith_junior_ml_contest.git
cd aith_junior_ml_contest

make install         # venv + базовые зависимости
make test            # unit-тесты
make pipeline-quick  # мини-EDA и обучение на demo-данных (CPU, ~30 c)
make demo            # веб-интерфейс на http://127.0.0.1:5050
```

`make demo` поднимает UI в эвристическом режиме — без GPU и без чекпойнта, чтобы
можно было сразу посмотреть продукт. Как получить или обучить ML-чекпойнт и
запустить полноценный инференс (`make serve`), описано в
[`docs/MODELS.md`](docs/MODELS.md).

## Полный пайплайн

Продакшен-конвейер разбит на семь шагов (`scripts/pipeline/01…07`) — сбор данных,
EDA, чистка и балансировка, VLM-разметка, обучение, оценка, отчёт:

```bash
cp config/pipeline.env.example .env   # параметры пайплайна
make pipeline                         # весь конвейер (нужны данные и GPU)
```

Частые команды:

| Действие | Команда |
|----------|---------|
| Быстрый прогон на demo (CPU) | `make pipeline-quick` |
| Полный пайплайн | `make pipeline` |
| Обучение (bash-скрипт) | `make train` |
| Обучение через Hydra + трекинг | `make train-cfg` |
| Оценка чекпойнтов | `make eval` |
| Веб-интерфейс с ML | `make serve` |
| Docker | `make docker` (UI на `http://127.0.0.1:8765`) |

## Результаты

Продакшен-модель (CLIP ViT-L-14 linear probe) на исследовательской валидации
(`n = 239`, сплит по `object_id`, без утечки сезонов):

| Метрика | Значение |
|---------|----------|
| Accuracy | 76.2% |
| macro-F1 | 61.0% |
| balanced macro-F1 (cap 35/класс) | 73.8% |

В валидации присутствуют все 6 классов. Лучше всего различаются `active_urban`
(F1 0.85) и `frozen_construction` (0.80), сложнее всего — `underused_infrastructure`
(0.38), что ожидаемо: этот класс визуально пересекается с соседними.

Как проект дошёл до этих цифр (главный рычаг — не архитектура, а качество меток):

| Этап | Accuracy |
|------|----------|
| OpenCLIP zero-shot | 0.20 |
| OpenCLIP fine-tune (OSM-метки) | 0.40 |
| CLIP probe на OSM-метках | 0.27 |
| **CLIP probe на VLM-метках (продакшен)** | **0.76** |
| CLIP probe, 3 макро-класса | 0.82 |

## MLOps

Обучение конфигурируется через **Hydra** (`conf/`), метрики пишутся в
**TensorBoard / W&B / MLflow** (единый трекер, любой бэкенд отключается), стадии
пайплайна описаны для **DVC** (`dvc.yaml`), качество кода держат **pre-commit +
ruff + black**.

```bash
make install-dev
python scripts/train.py model=vit_b32 data=demo tracking=tensorboard
make board   # дашборд на http://localhost:6006
```

Подробнее — в [`docs/MLOPS.md`](docs/MLOPS.md).

## Структура репозитория

```
├── Makefile                 # единая точка входа для всех команд
├── conf/                    # Hydra-конфиги (data / model / train / tracking)
├── config/                  # pipeline.env.example
├── dvc.yaml                 # стадии пайплайна для DVC
├── pyproject.toml           # настройки ruff / black / pytest
├── data/demo/               # мини-датасет для CI и pipeline-quick
├── web_app/                 # Flask UI + ML-инференс
├── scripts/
│   ├── train.py             # Hydra-точка входа обучения
│   ├── pipeline/            # шаги 01–07 продакшен-конвейера
│   ├── data_collection/     # сбор данных (OSM + Яндекс.Панорамы)
│   ├── data_quality/        # EDA, чистка, VLM-разметка
│   ├── training/            # CLIP probe, ансамбли, трекинг
│   ├── eval/                # бенчмарки, сравнение с GPKG
├── tests/                   # pytest (запускается в CI)
├── docs/                    # архитектура, EDA, метрики, продукт, MLOps
├── presentation/            # презентация для конкурса
└── submission/              # материалы для формы JMLC
```

## Соответствие критериям JMLC

- **Разработка и инженерия** — модульный конвейер, `Makefile`, Docker, CI,
  MLOps-стек (Hydra, DVC, трекинг).
- **Data Science** — EDA и аудит данных, чистка, VLM-переразметка, честная
  валидация со сплитом по `object_id` ([`docs/EDA.md`](docs/EDA.md),
  [`docs/METRICS.md`](docs/METRICS.md)).
- **Применение ИИ** — CLIP linear probe, VLM weak supervision (`scripts/agent/`).
- **Продуктовое мышление** — веб-инструмент для аналитиков с правками и экспортом
  в GPKG, цикл обратной связи ([`docs/PRODUCT.md`](docs/PRODUCT.md)).
- **Мотивация** — сопроводительные документы в [`submission/`](submission/).

Разбор по каждому критерию — в [`docs/JMLC_CRITERIA.md`](docs/JMLC_CRITERIA.md).

## Подача на конкурс

Чеклист и сроки — в [`SUBMISSION.md`](SUBMISSION.md), материалы формы — в
[`submission/`](submission/).

## Контакты

Дмитрий Новичков · novichkovde@pik.ru · [github.com/mfclabber](https://github.com/mfclabber)
