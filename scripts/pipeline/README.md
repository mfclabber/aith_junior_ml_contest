# Pipeline scripts

См. [`docs/PIPELINE.md`](../docs/PIPELINE.md).

| Script | Step |
|--------|------|
| `run_production.sh` | Master: 01 → 07 |
| `run_quick.sh` | CI / demo без полного data/ |
| `01_collect_data.sh` | OSM + Яндекс |
| `02_eda_audit.sh` | Аудит |
| `03_clean_and_balance.sh` | Чистка + crops + balance |
| `04_vlm_relabel.sh` | VLM (optional) |
| `05_train_probe.sh` | CLIP linear probe |
| `06_evaluate.sh` | Benchmark + GPKG |
| `07_report.sh` | Отчёт |
