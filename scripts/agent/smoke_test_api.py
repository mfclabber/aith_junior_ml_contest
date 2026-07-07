#!/usr/bin/env python3
"""Smoke-test классификаторов без поднятия Flask."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    errors: list[str] = []
    from PIL import Image

    val_line = (ROOT / "data/vlm_clean6/val.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(val_line)

    from web_app import probe_classifier as pc

    if not pc.checkpoints_available():
        errors.append("probe checkpoint missing")
    else:
        m = pc.get_probe_classifier()
        assert m is not None
        img = Image.open(row["image"]).convert("RGB")
        r = m.predict_image(img)
        if r["class_key"] not in m.class_mapping:
            errors.append(f"probe bad class {r}")
        print("probe:", r["class_key"], f"{r['confidence']:.3f}")

    from web_app import vlm_classifier as vc

    if not vc.checkpoints_available():
        errors.append("vlm checkpoint missing")
    else:
        try:
            v = vc.get_vlm()
            if v is None:
                errors.append("vlm load failed")
            else:
                img = Image.open(row["image"]).convert("RGB")
                r = v.predict_image(img)
                print("vlm:", r["class_key"], r.get("evidence_bbox_xyxy_norm"))
        except Exception as exc:
            errors.append(f"vlm oom/skip: {exc}")

    from web_app import ml_classifier as mc

    print("ensemble available:", mc.checkpoints_available())

    from web_app.classify_service import _active_classifier

    clf, backend = _active_classifier()
    print("active backend:", backend)

    if errors:
        print("FAIL:", errors, file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
