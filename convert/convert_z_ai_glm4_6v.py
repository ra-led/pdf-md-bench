#!/usr/bin/env python3
"""Normalize GLM-4.6v Markdown outputs to match the canonical format."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


INPUT_ROOT = Path("input/z-ai__glm-4_6v")
OUTPUT_ROOT = Path("output/z-ai__glm-4_6v")

BOX_PATTERN = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)
PAGE_PATTERN = re.compile(r"page_(\d+)\.md$", re.IGNORECASE)


def clean_content(raw_text: str) -> str:
    normalized = raw_text.replace("\r\n", "\n")
    matches = BOX_PATTERN.findall(normalized)
    if matches:
        cleaned_segments = [segment.strip() for segment in matches if segment.strip()]
        cleaned = "\n\n".join(cleaned_segments)
    else:
        cleaned = normalized.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def convert_document(doc_dir: Path) -> None:
    page_files = []
    for path in doc_dir.glob("page_*.md"):
        match = PAGE_PATTERN.search(path.name)
        if not match:
            continue
        page_index = int(match.group(1))
        page_files.append((page_index, path))

    if not page_files:
        return

    page_files.sort()
    output_dir = OUTPUT_ROOT / doc_dir.name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_pages: list[str] = []
    for page_index, page_path in page_files:
        raw_text = page_path.read_text(encoding="utf-8")
        cleaned = clean_content(raw_text)
        combined_pages.append(cleaned)
        output_path = output_dir / f"page_{page_index:03d}.md"
        output_path.write_text((cleaned + "\n") if cleaned else "\n", encoding="utf-8")

    combined_text = "\n\n".join(text for text in combined_pages if text)
    combined_output = output_dir / f"{doc_dir.name}.md"
    combined_output.write_text((combined_text + "\n") if combined_text else "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for doc_dir in sorted(INPUT_ROOT.iterdir()):
        if doc_dir.is_dir():
            convert_document(doc_dir)


if __name__ == "__main__":
    main()
