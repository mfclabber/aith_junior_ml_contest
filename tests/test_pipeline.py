"""Unit tests for JMLC repo (CI, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_metadata_valid():
    meta = json.loads((ROOT / "data/demo/metadata_demo.json").read_text())
    assert len(meta["train_samples"]) >= 4
    assert len(meta["val_samples"]) >= 2
    assert len(meta["class_mapping"]) == 6


def test_demo_images_exist():
    meta = json.loads((ROOT / "data/demo/metadata_demo.json").read_text())
    for split in ("train_samples", "val_samples"):
        for s in meta[split]:
            p = ROOT / "data/demo" / s["image_path"]
            assert p.is_file(), f"missing {p}"


def test_taxonomy_six_classes():
    from web_app.taxonomy import UTT_LABELS

    assert len(UTT_LABELS) == 6
    assert "natural_areas" in UTT_LABELS
    assert "active_urban" in UTT_LABELS


def test_heuristic_classify():
    from web_app.gpkg_io import classify_from_description

    cls, sub = classify_from_description("лесопарк чистая сцена")
    assert "Природ" in cls or cls


def test_refine_subclass_greenery():
    from web_app.taxonomy import refine_subclass

    sub = refine_subclass(
        "natural_areas",
        "просто зелень (деревья во дворах и тд)",
        "Парки и скверы",
    )
    assert sub


def test_pipeline_config_example_exists():
    assert (ROOT / "config/pipeline.env.example").is_file()


def test_flask_app_import():
    from web_app.app import app

    assert app is not None
