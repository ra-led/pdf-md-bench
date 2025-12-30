#!/usr/bin/env python3
"""Convert PP-Structure V3 markdown outputs to the standardized format."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


INPUT_ROOT = Path("input/pp_sv3")
OUTPUT_ROOT = Path("output/pp_v3")

PAGE_FILE_RE = re.compile(r"_(?P<index>\d+)\.md$", re.IGNORECASE)
DIV_PATTERN = re.compile(r"<div style=\"text-align: center;\">\s*(.*?)\s*</div>", re.DOTALL | re.IGNORECASE)
IMG_TAG_RE = re.compile(r"<img[^>]*src=\"([^\"]+)\"[^>]*>", re.IGNORECASE)
TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
BBOX_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)(?:\.[^._]+)?$")


def _extract_bbox_from_src(src: str) -> str | None:
    basename = Path(src).name
    stem = Path(basename).stem
    match = BBOX_RE.search(stem)
    if match:
        return " ".join(match.groups())
    digits = re.findall(r"\d+", stem)
    if len(digits) >= 4:
        return " ".join(digits[-4:])
    return None


def _convert_image_block(inner_html: str) -> str:
    img_match = IMG_TAG_RE.search(inner_html)
    if not img_match:
        return ""
    src = img_match.group(1)
    bbox = _extract_bbox_from_src(src)
    if not bbox:
        return ""
    return f'<img data-bbox="{bbox}">Image</img>'


def _convert_table_block(inner_html: str) -> str:
    table_match = TABLE_RE.search(inner_html)
    if not table_match:
        return inner_html.strip()
    table_html = table_match.group(0)
    return table_html.strip()


def _replace_div_blocks(text: str) -> str:
    def _replacement(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        lowered = inner.lower()
        if "<img" in lowered:
            return _convert_image_block(inner)
        if "<table" in lowered:
            return _convert_table_block(inner)
        return inner

    return DIV_PATTERN.sub(_replacement, text)


def convert_page(content: str) -> str:
    normalized = content.replace("\r\n", "\n")
    normalized = _replace_div_blocks(normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def convert_document(doc_dir: Path) -> None:
    md_files = []
    for path in doc_dir.glob("*.md"):
        match = PAGE_FILE_RE.search(path.name)
        if match:
            md_files.append((int(match.group("index")), path))
    if not md_files:
        return
    md_files.sort()

    output_dir = OUTPUT_ROOT / doc_dir.name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_pages: list[str] = []
    for index, path in md_files:
        converted = convert_page(path.read_text(encoding="utf-8"))
        combined_pages.append(converted)
        (output_dir / f"page_{index:03d}.md").write_text((converted + "\n") if converted else "\n", encoding="utf-8")

    combined = "\n\n".join(text for text in combined_pages if text)
    (output_dir / f"{doc_dir.name}.md").write_text((combined + "\n") if combined else "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for doc_dir in sorted(INPUT_ROOT.iterdir()):
        if doc_dir.is_dir():
            convert_document(doc_dir)


if __name__ == "__main__":
    main()
