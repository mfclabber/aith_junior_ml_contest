#!/usr/bin/env python3
"""Собрать PDF заявки из Markdown-источников (нужен pandoc + xelatex)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sources"
PDF = ROOT / "pdf"

DOCS = [
    ("CV.md", "CV_Новичков_Дмитрий.pdf", "Резюме"),
    ("CV_EXTENDED.md", "CV_EXTENDED_Новичков_Дмитрий.pdf", "Расширенное резюме"),
    ("MOTIVATION_LETTER.md", "MOTIVATION_LETTER_Новичков_Дмитрий.pdf", "Мотивационное письмо"),
    ("PROJECT_DESCRIPTION.md", "PROJECT_DESCRIPTION.pdf", "Описание проекта"),
]


def build(md: Path, pdf: Path, title: str) -> bool:
    cmd = [
        "pandoc", str(md), "-o", str(pdf),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=2cm",
        "-V", "fontsize=11pt",
        "-V", f"title={title}",
        "-V", "lang=ru",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ok  {pdf.name}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  skip {md.name}: {exc}")
        return False


def main() -> int:
    PDF.mkdir(parents=True, exist_ok=True)
    built = sum(build(SRC / s, PDF / o, t) for s, o, t in DOCS if (SRC / s).is_file())
    if not built:
        print("Нужен pandoc + xelatex. Готовые PDF уже лежат в submission/pdf/.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
