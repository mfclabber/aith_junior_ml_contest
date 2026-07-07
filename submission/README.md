# Материалы для формы JMLC

| Файл | Описание |
|------|----------|
| `pdf/CV_Новичков_Дмитрий.pdf` | Резюме |
| `pdf/CV_EXTENDED_Новичков_Дмитрий.pdf` | Расширенное резюме |
| `pdf/MOTIVATION_LETTER_Новичков_Дмитрий.pdf` | Мотивационное письмо |
| `pdf/PROJECT_DESCRIPTION.pdf` | Описание проекта (≤5 стр.) |

Исходники в Markdown — в `sources/`. Презентация — `../presentation/JMLC_pitch.pptx`.

Пересобрать PDF из Markdown (нужен `pandoc` + `xelatex`):

```bash
python generate_pdfs.py
```
