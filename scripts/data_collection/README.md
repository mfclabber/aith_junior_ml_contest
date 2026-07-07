# Data Collection Scripts

Scripts for collecting panorama dataset from OpenStreetMap and Yandex Maps.

## Scripts

- **pano.py** - Download single panorama from Yandex Maps
- **dataset_collector.py** - Main script for collecting dataset by UTT taxonomy classes from OSM
- **download_all_seasons.sh** - Batch script to download all classes with all seasons

## Usage

### Download all classes with all seasons

```bash
bash scripts/data_collection/download_all_seasons.sh
```

### Single class collection

```bash
python3 scripts/data_collection/dataset_collector.py \
    --class natural_areas \
    --max-results 100 \
    --collect-seasons \
    --download \
    --output-dir data/dataset
```

### Single panorama download

```bash
python3 scripts/data_collection/pano.py \
    -c 55.7558,37.6173 \
    -o output.jpg \
    -z 1
```

