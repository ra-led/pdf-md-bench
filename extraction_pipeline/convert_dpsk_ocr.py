#!/usr/bin/env python3
"""Convert DeepSeek OCR detection output to Markdown format used in ./conon."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


INPUT_ROOT = Path("input/dpsk-ocr-Gundam")
OUTPUT_ROOT = Path("output/dpsk-ocr-Gundam")

PAGE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)*<--- Page Split --->(?:\r?\n)*")
TAG_PATTERN = re.compile(
    r"<\|ref\|>(?P<ref>.+?)<\|/ref\|><\|det\|>\s*\[\[(?P<bbox>[^\]]+)\]\]\s*<\|/det\|>",
    flags=re.DOTALL,
)


def _normalize_bbox(raw_bbox: str) -> str:
    parts = [value.strip() for value in raw_bbox.replace("[", "").replace("]", "").split(",")]
    return " ".join(filter(None, parts))


def _strip_tags(content: str) -> str:
    def _replacer(match: re.Match[str]) -> str:
        ref_type = match.group("ref").strip().lower()
        bbox = _normalize_bbox(match.group("bbox"))
        if ref_type == "image" and bbox:
            return f'<img data-bbox="{bbox}"></img>\n'
        return ""

    cleaned = TAG_PATTERN.sub(_replacer, content)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_pages(raw_text: str) -> list[str]:
    if not raw_text.strip():
        return []
    segments = PAGE_SPLIT_PATTERN.split(raw_text.strip())
    return segments if segments else []


def convert_document(det_file: Path) -> None:
    raw_text = det_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    pages_raw = _split_pages(raw_text)
    if not pages_raw:
        pages_raw = [raw_text]

    output_dir = OUTPUT_ROOT / det_file.parent.name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_outputs: list[str] = []
    for index, page in enumerate(pages_raw):
        converted = _strip_tags(page)
        page_outputs.append(converted)
        page_path = output_dir / f"page_{index:03d}.md"
        page_path.write_text((converted + "\n") if converted else "\n", encoding="utf-8")

    combined_path = output_dir / f"{det_file.parent.name}.md"
    combined_content = "\n\n".join(text for text in page_outputs if text)
    combined_path.write_text((combined_content + "\n") if combined_content else "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    det_files = sorted(path for path in INPUT_ROOT.rglob("*_det.mmd") if path.is_file())
    for det_file in det_files:
        convert_document(det_file)


if __name__ == "__main__":
    main()
