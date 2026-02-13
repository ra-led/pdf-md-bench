#!/usr/bin/env python3
"""End-to-end pipeline (tables only): PDF -> DeepSeek OCR det.mmd -> table blocks -> extracted JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

from ocr_runner import run_ocr_with_deepseek_runner
from openrouter_stream import call_openrouter_json_streaming

try:
    from dedupe_repetitions import fairy_tame as _fairy_tame  # type: ignore
except Exception:
    def _fairy_tame(text: str) -> str:
        return text


IE_SYSTEM_PROMPT = "You are a precise information extraction system."

TABLE_IE_PROMPT_TEMPLATE = """You extract structured characteristics ONLY from the provided table fragment.

Input: one table (HTML or Markdown) from a technical questionnaire.
Output: a single VALID JSON array only. No markdown, no comments.

JSON format:
[
  {{
    "name": "<parameter name or full requirement wording>",
    "class": "<top-level section name if known; otherwise infer from table context>",
    "request_value": "<value requested by customer>",
    "islink": <true/false>
  }},
  ...
]

STRICT RULES:
1) Output ONLY valid JSON (array at top-level). No extra text.
2) Use ONLY information present in this table. Do not use outside context.
3) Preserve units, references, signs, numeric formatting.
4) If row has requirement text without explicit value, use "Да" unless table says otherwise.
5) islink=true if name/request_value references external docs/tables/standards (ГОСТ, СП, ПУЭ, Приложение, "по таблице", "согласно", etc.).
6) Do not invent missing rows.

TABLE:
<<<TABLE
{table_text}
TABLE>>>
"""

SPLIT_CHECK_PROMPT_TEMPLATE = """Decide whether two characteristics are actually one requirement split across table/page boundary.

Return STRICT JSON object only.

Previous table last characteristic:
{prev_item}

Next table first characteristic:
{next_item}

Rules:
1) If second item clearly continues/completes the first (broken sentence, continued value/reference, unfinished thought), set is_split=true.
2) If they are independent rows/requirements, set is_split=false.
3) If is_split=true, provide merged_item with properly combined fields:
   - name: full merged wording
   - class: most appropriate class
   - request_value: merged value text
   - islink: true if either side has a link/reference
4) If is_split=false, still provide merged_item as a best-effort combination but it will be ignored.
"""

IE_RESPONSE_SCHEMA = {
    "name": "ie_extraction_list",
    "strict": True,
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "class": {"type": "string"},
                "request_value": {"type": "string"},
                "islink": {"type": "boolean"},
            },
            "required": ["name", "class", "request_value", "islink"],
            "additionalProperties": False,
        },
    },
}

SPLIT_CHECK_SCHEMA = {
    "name": "split_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_split": {"type": "boolean"},
            "merged_item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "class": {"type": "string"},
                    "request_value": {"type": "string"},
                    "islink": {"type": "boolean"},
                },
                "required": ["name", "class", "request_value", "islink"],
                "additionalProperties": False,
            },
        },
        "required": ["is_split", "merged_item"],
        "additionalProperties": False,
    },
}

FENCE_LINE_RE = re.compile(r'^[\'\"]?```.*$', re.MULTILINE)
PAGE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)*<--- Page Split --->(?:\r?\n)*")
TAG_PATTERN = re.compile(
    r"<\|ref\|>(?P<ref>.+?)<\|/ref\|><\|det\|>\s*\[\[(?P<bbox>[^\]]+)\]\]\s*<\|/det\|>",
    flags=re.DOTALL,
)
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", flags=re.IGNORECASE | re.DOTALL)

def clean_input_text(text: str) -> str:
    text = FENCE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_from_single_table(
    *,
    table_text: str,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> list[dict]:
    prompt = TABLE_IE_PROMPT_TEMPLATE.format(table_text=table_text)
    parsed = call_openrouter_json_streaming(
        api_key=api_key,
        model=model,
        system_prompt=IE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=IE_RESPONSE_SCHEMA,
        temperature=0.0,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        require_parameters=require_parameters,
        log_label="[LLM]",
    )
    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected list from table extraction, got {type(parsed).__name__}")
    return parsed


def check_split_between_tables(
    *,
    prev_item: dict,
    next_item: dict,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> tuple[bool, dict]:
    prompt = SPLIT_CHECK_PROMPT_TEMPLATE.format(
        prev_item=json.dumps(prev_item, ensure_ascii=False, indent=2),
        next_item=json.dumps(next_item, ensure_ascii=False, indent=2),
    )
    parsed = call_openrouter_json_streaming(
        api_key=api_key,
        model=model,
        system_prompt=IE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=SPLIT_CHECK_SCHEMA,
        temperature=0.0,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        require_parameters=require_parameters,
        log_label="[LLM]",
    )
    if not isinstance(parsed, dict) or "is_split" not in parsed or "merged_item" not in parsed:
        raise RuntimeError("Invalid split check response")
    return bool(parsed["is_split"]), parsed["merged_item"]


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


def _extract_markdown_table_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue

        start = i
        j = i
        while j < len(lines) and ("|" in lines[j] or not lines[j].strip()):
            j += 1

        chunk_lines = lines[start:j]
        chunk_text = "\n".join(chunk_lines).strip()

        has_separator = False
        for ln in chunk_lines:
            compact = ln.replace(" ", "")
            if re.search(r"\|?:?-{3,}:?\|", compact):
                has_separator = True
                break

        if has_separator and chunk_text:
            blocks.append(chunk_text)

        i = j

    return blocks


def extract_tables_from_det(det_file: Path) -> list[str]:
    raw_text = det_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    pages_raw = _split_pages(raw_text) or [raw_text]
    cleaned_doc = "\n\n".join(_strip_tags(page) for page in pages_raw if page.strip())
    cleaned_doc = clean_input_text(_fairy_tame(cleaned_doc))

    tables: list[str] = []

    # HTML tables in document order.
    for match in HTML_TABLE_RE.finditer(cleaned_doc):
        block = match.group(0).strip()
        if block:
            tables.append(block)

    # Markdown tables can appear without HTML tags.
    markdown_tables = _extract_markdown_table_blocks(cleaned_doc)
    for block in markdown_tables:
        if block not in tables:
            tables.append(block)

    return tables


def merge_cross_table_splits(
    table_results: list[list[dict]],
    *,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> list[list[dict]]:
    for i in range(len(table_results) - 1):
        print(f"[STEP] Boundary check between table {i + 1} and table {i + 2}")
        current = table_results[i]
        nxt = table_results[i + 1]
        if not current or not nxt:
            continue

        is_split, merged_item = check_split_between_tables(
            prev_item=current[-1],
            next_item=nxt[0],
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_tokens=max_tokens,
            require_parameters=require_parameters,
        )

        if is_split:
            print(f"[INFO] Table boundary {i + 1}/{i + 2}: merged split characteristic")
            current[-1] = merged_item
            del nxt[0]
        else:
            print(f"[INFO] Table boundary {i + 1}/{i + 2}: kept as separate characteristics")

    return table_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Input PDF file path")
    parser.add_argument("--out_json", required=True, help="Output JSON path")

    parser.add_argument("--openrouter_api_key", default=None, help="OpenRouter API key")
    parser.add_argument("--openrouter_model", default="deepseek/deepseek-v3.2")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument("--max_tokens", type=int, default=30000)
    parser.add_argument("--no_require_parameters", action="store_true")

    parser.add_argument(
        "--dpsk_workdir",
        default=os.environ.get("DPSK_WORKDIR"),
        help="Path to DeepSeek-OCR-vllm folder containing run_dpsk_ocr_pdf.py",
    )
    parser.add_argument("--dpsk_run_script", default="run_dpsk_ocr_pdf.py")
    parser.add_argument("--dpsk_model_path", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--python_bin", default=sys.executable)

    parser.add_argument("--det_file", default=None, help="Skip OCR and use existing *_det.mmd file")
    parser.add_argument("--workdir", default=None, help="Intermediate workdir; default is a temp directory")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--save_tables_dir", default=None, help="Optional directory to save extracted table blocks")

    args = parser.parse_args()
    print("[STEP] Starting tables-only pipeline")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OpenRouter API key. Pass --openrouter_api_key or set OPENROUTER_API_KEY")

    if args.workdir:
        base_workdir = Path(args.workdir)
        base_workdir.mkdir(parents=True, exist_ok=True)
        cleanup: Callable[[], None] = lambda: None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="pdf2md_tables_pipeline_")
        base_workdir = Path(temp_dir.name)
        cleanup = temp_dir.cleanup

    try:
        raw_ocr_root = base_workdir / "raw_ocr"
        print(f"[INFO] Intermediate workdir: {base_workdir}")

        if args.det_file:
            det_file = Path(args.det_file)
            if not det_file.exists():
                raise SystemExit(f"det file not found: {det_file}")
            print(f"[STEP] Using existing det file: {det_file}")
        else:
            if not args.dpsk_workdir:
                raise SystemExit("Pass --dpsk_workdir for OCR run or provide --det_file to skip OCR")
            det_file = run_ocr_with_deepseek_runner(
                pdf_path=pdf_path,
                dpsk_workdir=Path(args.dpsk_workdir),
                raw_ocr_root=raw_ocr_root,
                dpsk_model_path=args.dpsk_model_path,
                python_bin=args.python_bin,
                run_script=args.dpsk_run_script,
            )
            print(f"[INFO] OCR det file: {det_file}")

        print("[STEP] Extracting table blocks from OCR output")
        tables = extract_tables_from_det(det_file)
        if not tables:
            raise SystemExit("No tables found in OCR output")
        print(f"[INFO] Tables found: {len(tables)}")

        save_tables_dir = Path(args.save_tables_dir) if args.save_tables_dir else None
        if save_tables_dir is not None:
            save_tables_dir.mkdir(parents=True, exist_ok=True)

        table_results: list[list[dict]] = []
        for idx, table in enumerate(tables):
            print(f"[STEP] Extracting table {idx + 1}/{len(tables)}")
            if save_tables_dir is not None:
                (save_tables_dir / f"table_{idx:03d}.md").write_text(table + "\n", encoding="utf-8")

            items = extract_from_single_table(
                table_text=table,
                api_key=api_key,
                model=args.openrouter_model,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                require_parameters=not args.no_require_parameters,
            )
            print(f"[INFO] Extracted characteristics from table {idx + 1}: {len(items)}")
            table_results.append(items)

        print("[STEP] Checking neighboring table boundaries for split characteristics")
        merged_table_results = merge_cross_table_splits(
            table_results,
            api_key=api_key,
            model=args.openrouter_model,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            require_parameters=not args.no_require_parameters,
        )

        final_items: list[dict] = []
        for items in merged_table_results:
            final_items.extend(items)
        print(f"[INFO] Final characteristics count: {len(final_items)}")

        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[STEP] Writing JSON output: {out_json_path}")
        out_json_path.write_text(json.dumps(final_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Done. JSON saved to: {out_json_path.resolve()}")
        print(f"Tables processed: {len(tables)}")

    finally:
        if not args.keep_intermediate:
            cleanup()


if __name__ == "__main__":
    main()
