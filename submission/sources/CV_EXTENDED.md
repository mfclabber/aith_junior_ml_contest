# Расширенное резюме

**Дмитрий Новичков**  
ML-инженер · ПИК AI Lab  
novichkovde@pik.ru · github.com/mfclabber

---

## 1. Образование

### Университет ИТМО
**Бакалавриат, «Робототехника и искусственный интеллект»**  
GPA: **4.4 / 5.0**

Ключевые направления: машинное обучение, компьютерное зрение, робототехника, теория управления, математический анализ, линейная алгебра, теория вероятностей.

**Выпускная квалификационная работа (ВКР):**  
«Применение Vision-Language-Action моделей и генеративных наград для обучения с подкреплением в робототехнике»

- **Трек 1:** развёртывание eval-пайплайна OpenVLA-OFT (7B) на бенчмарке LIBERO; success rate **96.5%** в среднем (100% Object, 93% Spatial).
- **Трек 2:** Text2Reward — LLM (Qwen2.5-VL) генерирует dense reward functions для PPO/SAC на MetaWorld MT10; SAC + LLM reward = **100% SR** на reach-v3 без ручного reward engineering.

### Beijing Institute of Technology (BIT)
**Академический обмен**  
Защита работы: «Foundation Models as World Models for Reinforcement Learning».

---

## 2. Профессиональный опыт

### ПИК AI Lab / ПИК Digital — ML-инженер → Senior ML Engineer
*2024 — настоящее время*

**Классификация городских участков (`classification_street_buildings`):**
- Построил end-to-end контур: OSM + Яндекс.Панорамы + GPKG → perspective crops → VLM.
- Провёл zero-shot бенчмарк Qwen3-VL-8B на 168 панорамах (macro IoU 0.197).
- Дообучил PaliGemma 3B (LoRA): macro IoU **0.517**, F1 **0.63**, evidence IoU **0.64**.
- Разработал Flask веб-пилот: карта, галерея панорам по граням участка, weak supervision через UI.
- Доклад на конференции **AIJ 2025**.

**Распознавание документов (`documents_worker`):**
- Production-сервис проверки 2-НДФЛ и паспортов: VLM/OCR, детекция дефектов (срез, размытие, отсутствие подписанта), compliance-правила.
- Docker-контейнеризация, API, веб-UI; интеграция в ипотечный процесс.
- Доклад **DataFest 2026**.

**Инфраструктура ML:**
- Распределённое обучение моделей 3B+ (FSDP, DeepSpeed); ускорение inference через TensorRT (3–5×).
- macro-F1 VLM на задаче городской застройки: рост с **0.71 до 0.86** на 12 классах (расширенный контур ПИК).

### Сколтех — ML Researcher
*июнь — сентябрь 2025*

- Исследования VLA и RL для манипуляции и locomotion.
- Reward design: семантическая + физическая компоненты награды.
- Eval OpenVLA на LIBERO, подготовка к RL-дообучению (SimpleVLA-RL).

### СБЕР — ML-инженер (робототехника)
*~2.5 года*

- Обучение locomotion policy для **Unitree Go1**: Gazebo, RaiSim, Isaac Sim → физический робот.
- Domain randomization, system identification, перенос policy Sim-to-Real.

### Formula Student — руководитель автономного стека (ИТМО)
*параллельно с учёбой и работой*

- ROS2, SLAM, MPC, перенос policy из симулятора на физический болид.
- Координация команды, интеграция perception + planning + control.

---

## 3. Навыки (детализация)

| Область | Инструменты и технологии |
|---------|--------------------------|
| Deep Learning | PyTorch, JAX, HuggingFace, DeepSpeed, FSDP, AMP, LoRA/PEFT |
| CV / VLM | OpenCLIP, PaliGemma, Qwen-VL, SAM, Grad-CAM, GeoPandas |
| NLP / GenAI | vLLM, TRL, RAG, structured extraction, prompt engineering |
| RL | Stable-Baselines3, RLlib, PPO, SAC, GRPO, Text2Reward |
| Robotics | ROS2, MuJoCo, Isaac Sim, LeRobot, OpenVLA |
| Production | Docker, Kubernetes, FastAPI, Flask, ONNX, TensorRT |
| Языки | Python (основной), C++, SQL, Go |

---

## 4. Достижения и публичная активность

- **AIJ 2025** — доклад «От панорамы к решению: VLM и классификация городской застройки»
- **DataFest 2026** — доклад «Генеративное распознавание документов в production»
- **Open-source:** github.com/mfclabber — `diploma`, `classification_street_buildings`, `documents_worker`, `fixik`, `hr_bot`
- **Formula Student ИТМО** — автономное вождение, Sim-to-Real
- **GPA 4.4** ИТМО, обмен BIT, исследования в Сколтехе

---

## 5. Мотивация развития (кратко)

Хочу углубить теоретическую базу в ML (generative models, RL, CV) и связать фундаментальные исследования с прикладными продуктами в urban AI и embodied intelligence. Магистратура **AI Talent Hub (ИТМО)** — логичное продолжение траектории: уже учился в ИТМО, работаю с промышленными задачами и хочу систематизировать знания под наставничеством программы.
