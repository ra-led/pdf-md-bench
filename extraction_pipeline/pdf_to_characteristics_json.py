#!/usr/bin/env python3
"""End-to-end pipeline: PDF -> DeepSeek OCR det.mmd -> Markdown -> extracted JSON."""

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

IE_PROMPT_TEMPLATE = """You extract structured characteristics and requirements from a technical questionnaire document.

Input: document text (converted from PDF; may contain headings, lists, and HTML tables).
Output: a single VALID JSON array only. No markdown, no comments.

JSON format:
[
  {{
    "name": "<parameter name or full requirement wording>",
    "class": "<top-level section name>",
    "request_value": "<value requested by customer>",
    "islink": <true/false>
  }},
  ...
]

STRICT RULES:
1) Output ONLY valid JSON (array at top-level). No extra text.
2) name:
   - Use the parameter name / requirement wording from the document.
   - Normalize whitespace and line breaks, but do NOT change meaning.
   - If a requirement is a full sentence (e.g., "... должен ..."), use the full requirement text as name in one line.
3) request_value:
   - For parameters defined in tables/structured blocks: use the value associated with that parameter in the document.
   - For declarative requirements that have no explicit value: use a confirming value such as "Да" (or "Подтверждаю" if explicitly present).
   - Keep values like "Определяет завод-изготовитель", "Уточняется...", "По таблице ...", "___", "____" as-is.
   - If a parameter has multiple sub-values, merge them into ONE string separated by "; " while preserving all numbers, units and references.
4) class:
   - Use the nearest relevant top-level section heading for the attribute.
   - If unclear, use the closest preceding section title.
5) islink:
   - true if name or request_value contains a reference to external documents/standards/appendices/tables (ГОСТ, СП, ПУЭ, Приложение, "по таблице …", "согласно …", etc.).
   - otherwise false.
6) Do NOT invent attributes that are not in the document.
7) Preserve units, minus signs (−), plus signs (+), decimal separators, and formatting as much as possible.

DOCUMENT TEXT:
<<<DOC
{doc_text}
DOC>>>
"""

IE_RESPONSE_SCHEMA = {
    "name": "ie_extraction_list",
    "strict": True,
    "schema": {
        "type": "array",
        "description": "List of extracted requirements from the document.",
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

FENCE_LINE_RE = re.compile(r'^[\'"]?```.*$', re.MULTILINE)
PAGE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)*<--- Page Split --->(?:\r?\n)*")
TAG_PATTERN = re.compile(
    r"<\|ref\|>(?P<ref>.+?)<\|/ref\|><\|det\|>\s*\[\[(?P<bbox>[^\]]+)\]\]\s*<\|/det\|>",
    flags=re.DOTALL,
)

def clean_input_text(text: str) -> str:
    text = FENCE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_characteristics(
    markdown_path: Path,
    out_json_path: Path,
    api_key: str,
    model: str,
    max_retries: int,
    timeout_s: int,
    max_tokens: int,
    require_parameters: bool,
) -> None:
    print(f"[STEP] Reading markdown for extraction: {markdown_path}")
    doc_text = _fairy_tame(clean_input_text(markdown_path.read_text(encoding="utf-8")))
    print(f"[INFO] Prepared extraction text length: {len(doc_text)} chars")
    prompt = IE_PROMPT_TEMPLATE.format(doc_text=doc_text)

    parsed = call_openrouter_json_streaming(
        api_key=api_key,
        model=model,
        system_prompt=IE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=IE_RESPONSE_SCHEMA,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        max_retries=max_retries,
        require_parameters=require_parameters,
        log_label="[LLM]",
    )
    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected top-level JSON array, got {type(parsed).__name__}")

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[STEP] Writing JSON output: {out_json_path}")
    out_json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def convert_det_to_markdown(det_file: Path, output_dir: Path) -> Path:
    raw_text = det_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    pages_raw = _split_pages(raw_text) or [raw_text]

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
    return combined_path


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

    args = parser.parse_args()
    print("[STEP] Starting full document pipeline")

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
        temp_dir = tempfile.TemporaryDirectory(prefix="pdf2md_pipeline_")
        base_workdir = Path(temp_dir.name)
        cleanup = temp_dir.cleanup

    try:
        raw_ocr_root = base_workdir / "raw_ocr"
        md_root = base_workdir / "markdown"
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

        print("[STEP] Converting OCR det output to markdown")
        markdown_path = convert_det_to_markdown(det_file=det_file, output_dir=md_root / pdf_path.stem)
        print(f"[INFO] Markdown path: {markdown_path}")

        print("[STEP] Running information extraction")
        extract_characteristics(
            markdown_path=markdown_path,
            out_json_path=Path(args.out_json),
            api_key=api_key,
            model=args.openrouter_model,
            max_retries=args.max_retries,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            require_parameters=not args.no_require_parameters,
        )

        print(f"Done. JSON saved to: {Path(args.out_json).resolve()}")
        print(f"Markdown used for extraction: {markdown_path.resolve()}")

    finally:
        if not args.keep_intermediate:
            cleanup()


if __name__ == "__main__":
    main()
