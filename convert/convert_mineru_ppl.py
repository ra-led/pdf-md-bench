#!/usr/bin/env python3
"""Convert MinerU prediction JSON into standardized Markdown outputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


INPUT_ROOT = Path("input/mineru_ppl")
OUTPUT_ROOT = Path("output/mineru_ppl")


@dataclass
class Block:
    order: int
    page_index: int
    bbox: List[float]
    content: Optional[str]


def _format_bbox(bbox: Iterable[float]) -> str:
    formatted: List[str] = []
    for value in bbox:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            formatted.append(str(value))
            continue
        if numeric.is_integer():
            formatted.append(str(int(numeric)))
        else:
            formatted.append(str(numeric))
    return " ".join(formatted)


def _format_text(item: dict) -> Optional[str]:
    text = (item.get("text") or "").strip()
    if not text:
        return None
    level = item.get("text_level")
    if level:
        try:
            level_int = int(level)
        except (TypeError, ValueError):
            level_int = 0
        if level_int > 0:
            level_int = max(1, min(6, level_int))
            prefix = "#" * level_int
            return f"{prefix} {text.strip()}"
    return text


def _format_image(item: dict) -> Optional[str]:
    bbox = item.get("bbox")
    if not bbox:
        return None
    caption_parts = []
    for key in ("image_caption", "image_footnote"):
        values = item.get(key) or []
        caption_parts.extend(part.strip() for part in values if isinstance(part, str) and part.strip())
    description = " ".join(caption_parts) if caption_parts else "Image"
    return f'<img data-bbox="{_format_bbox(bbox)}">{description}</img>'


def _format_table(item: dict) -> Optional[str]:
    table_body = (item.get("table_body") or "").strip()
    return table_body or None


FORMATTERS = {
    "text": _format_text,
    "image": _format_image,
    "table": _format_table,
    "equation": _format_text,
}


def load_blocks(json_path: Path) -> List[Block]:
    with json_path.open("r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    blocks: List[Block] = []
    for order, item in enumerate(raw_items):
        block_type = item.get("type")
        if block_type not in FORMATTERS:
            continue
        formatter = FORMATTERS[block_type]
        content = formatter(item)
        if not content:
            continue
        bbox = item.get("bbox") or [0, 0, 0, 0]
        page_idx = int(item.get("page_idx", 0))
        blocks.append(Block(order=order, page_index=page_idx, bbox=bbox, content=content))

    blocks.sort(key=lambda blk: (blk.page_index, blk.bbox[1] if len(blk.bbox) > 1 else 0, blk.bbox[0] if blk.bbox else 0, blk.order))
    return blocks


def write_pages(blocks: List[Block], output_dir: Path, doc_name: str) -> None:
    pages: dict[int, List[str]] = {}
    for block in blocks:
        pages.setdefault(block.page_index, []).append(block.content.strip())

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined: List[str] = []
    for page_idx in sorted(pages):
        content = "\n\n".join(filter(None, pages[page_idx]))
        combined.append(content)
        (output_dir / f"page_{page_idx:03d}.md").write_text((content + "\n") if content else "\n", encoding="utf-8")

    combined_text = "\n\n".join(text for text in combined if text)
    (output_dir / f"{doc_name}.md").write_text((combined_text + "\n") if combined_text else "\n", encoding="utf-8")


def convert_document(doc_dir: Path) -> None:
    auto_dir = doc_dir / "auto"
    if not auto_dir.exists():
        return
    json_files = list(auto_dir.glob("*_content_list.json"))
    if not json_files:
        return
    json_path = json_files[0]
    blocks = load_blocks(json_path)
    if not blocks:
        return
    output_dir = OUTPUT_ROOT / doc_dir.name
    write_pages(blocks, output_dir, doc_dir.name)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for doc_dir in sorted(INPUT_ROOT.iterdir()):
        if doc_dir.is_dir():
            convert_document(doc_dir)


if __name__ == "__main__":
    main()
