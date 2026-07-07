# Резюме (CV)

**Дмитрий Новичков**  
ML-инженер · ПИК AI Lab  
Email: novichkovde@pik.ru · GitHub: github.com/mfclabber

---

## Профиль

ML-инженер с опытом в computer vision, vision-language models и production ML. Совмещаю исследовательскую работу (VLA, RL, embodied AI) с промышленной разработкой: от сбора данных и обучения моделей до веб-сервисов и интеграции в бизнес-процессы.

---

## Опыт работы

**ПИК AI Lab / ПИК Digital** — ML-инженер (Senior)  
*2024 — н.в.*

- Классификация городских участков по уличным панорамам: пайплайн OSM → Yandex panoramas → VLM (Qwen3-VL, PaliGemma LoRA), веб-пилот на Flask.
- Генеративное распознавание документов (2-НДФЛ, паспорт): VLM/OCR, детекция дефектов, compliance-правила, Docker-сервис в production.
- Дообучение VLM на крупных датасетах (50K+ изображений), оптимизация обучения (FSDP/DeepSpeed) и inference (TensorRT).

**Сколтех** — ML Researcher  
*июнь — сентябрь 2025*

- VLA-модели и reinforcement learning для робототехники: reward design, curriculum learning, eval-пайплайн OpenVLA на LIBERO.

**СБЕР** — ML-инженер (робототехника)  
*~2.5 года*

- Locomotion policy для Unitree Go1: обучение в симуляторе, Sim-to-Real, domain randomization.

---

## Образование

**Университет ИТМО** — бакалавриат «Робототехника и искусственный интеллект»  
GPA 4.4 / 5.0

**Beijing Institute of Technology** — академический обмен  
Дипломная работа: «Foundation Models as World Models for Reinforcement Learning»

**ВКР (ИТМО):** VLA-модели (OpenVLA на LIBERO) + Text2Reward (LLM-генерация reward functions для RL на MetaWorld).

---

## Ключевые навыки

**ML/CV:** PyTorch, HuggingFace, OpenCLIP, VLM (PaliGemma, Qwen-VL), LoRA/PEFT, weak supervision  
**RL / Robotics:** Stable-Baselines3, MuJoCo, Isaac Sim, ROS2, Sim-to-Real  
**MLOps / Backend:** Docker, FastAPI, Flask, GeoPandas, ONNX, TensorRT  
**Языки:** Python, C++, SQL

---

## Проекты (выборочно)

| Проект | Суть | Результат |
|--------|------|-----------|
| `classification_street_buildings` | VLM-классификация участков по панорамам | IoU 0.517 после FT; веб-пилот |
| `documents_worker` | OCR + дефекты документов в production | ~21 с на документ, API + UI |
| `diploma` | VLA eval + RL с LLM rewards | 96.5% SR на LIBERO; 100% reach MetaWorld |
| Formula Student ИТМО | Автономный стек болида | ROS2, SLAM, MPC, Sim-to-Real policy |

---

## Достижения

- Доклад **AIJ 2025** — VLM для оценки городской застройки  
- Доклад **DataFest 2026** — генеративное распознавание документов в production  
- Руководство командой **Formula Student** (ИТМО) — автономное вождение
