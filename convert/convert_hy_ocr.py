#!/usr/bin/env python3
"""Convert Hanyan OCR Markdown output to the target Markdown format."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


INPUT_ROOT = Path("input/hy_ocr")
OUTPUT_ROOT = Path("output/hy_ocr")

PAGE_SPLIT_PATTERN = re.compile(r"---\s*\n\s*<!-- Page \d+ -->\s*\n", flags=re.MULTILINE)
TABLE_SEPARATOR_RE = re.compile(r"^:?[- ]{2,}:?$")


def split_pages(raw_text: str) -> list[str]:
    cleaned = raw_text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in PAGE_SPLIT_PATTERN.split(cleaned) if part.strip()]
    return parts if parts else [cleaned]


def table_lines_to_html(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        trimmed = stripped.strip("|")
        cells = [cell.strip() for cell in trimmed.split("|")]
        if cells and all(TABLE_SEPARATOR_RE.match(cell.replace(" ", "")) for cell in cells if cell != ""):
            continue
        rows.append(cells)

    if not rows:
        return "\n".join(lines)

    row_html: list[str] = ["<table>"]
    for row in rows:
        row_html.append("  <tr>")
        for cell in row:
            row_html.append(f"    <td>{cell}</td>")
        row_html.append("  </tr>")
    row_html.append("</table>")
    return "\n".join(row_html)


def convert_tables(page_text: str) -> str:
    lines = page_text.splitlines()
    output: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.strip().startswith("|"):
            table_block: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_block.append(lines[idx])
                idx += 1
            output.append(table_lines_to_html(table_block))
        else:
            output.append(line.rstrip())
            idx += 1
    converted = "\n".join(output)
    converted = re.sub(r"\n{3,}", "\n\n", converted)
    return converted.strip()


def convert_document(md_file: Path) -> None:
    raw_text = md_file.read_text(encoding="utf-8")
    pages = split_pages(raw_text)
    output_dir = OUTPUT_ROOT / md_file.stem
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_pages: list[str] = []
    for index, page in enumerate(pages):
        converted = convert_tables(page)
        combined_pages.append(converted)
        page_path = output_dir / f"page_{index:03d}.md"
        page_path.write_text((converted + "\n") if converted else "\n", encoding="utf-8")

    combined_path = output_dir / f"{md_file.stem}.md"
    combined_content = "\n\n".join(text for text in combined_pages if text)
    combined_path.write_text((combined_content + "\n") if combined_content else "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    md_files = sorted(path for path in INPUT_ROOT.glob("*.md") if path.is_file())
    for md_file in md_files:
        convert_document(md_file)


if __name__ == "__main__":
    main()
