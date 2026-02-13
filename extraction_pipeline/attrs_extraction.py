import argparse
import json
import os
import re
from pathlib import Path
import time
from tqdm import tqdm

import requests

from dedupe_repetitions import fairy_tame

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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
   - If a requirement is a full sentence (e.g., “... должен ...”), use the full requirement text as name in one line.
3) request_value:
   - For parameters defined in tables/structured blocks: use the value associated with that parameter in the document.
   - For declarative requirements that have no explicit value: use a confirming value such as "Да" (or "Подтверждаю" if explicitly present).
   - Keep values like "Определяет завод-изготовитель", "Уточняется...", "По таблице ...", "___", "____" as-is.
   - If a parameter has multiple sub-values, merge them into ONE string separated by "; " while preserving all numbers, units and references.
4) class:
   - Use the nearest relevant top-level section heading for the attribute.
   - If unclear, use the closest preceding section title.
5) islink:
   - true if name or request_value contains a reference to external documents/standards/appendices/tables (ГОСТ, СП, ПУЭ, Приложение, “по таблице …”, “согласно …”, etc.).
   - otherwise false.
6) Do NOT invent attributes that are not in the document.
7) Preserve units, minus signs (−), plus signs (+), decimal separators, and formatting as much as possible.

DOCUMENT TEXT:
<<<DOC
{doc_text}
DOC>>>
"""

# JSON Schema for OpenRouter Structured Outputs: list of strict objects
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

def clean_input_text(text: str) -> str:
    # Remove full fence lines
    text = FENCE_LINE_RE.sub("", text)

    # Optional: collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()



def call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 20000,
    timeout_s: int = 180,
    max_retries: int = 5,
    require_parameters: bool = True,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": IE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": IE_RESPONSE_SCHEMA,
        },
    }

    if require_parameters:
        payload["provider"] = {"require_parameters": True}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
            if r.status_code >= 400:
                try:
                    err = r.json()
                except Exception:
                    err = {"error": r.text}
                raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)[:2000]}")

            j = r.json()
            content = j["choices"][0]["message"]["content"]

            # If provider already returns parsed JSON, accept list directly.
            if isinstance(content, list):
                result = content
            else:
                # Some providers return content blocks; join text parts.
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                if not isinstance(content, str):
                    content = str(content)
                result = json.loads(content)

            # Sanity check (optional but helpful)
            if not isinstance(result, list):
                raise ValueError(f"Expected top-level JSON array, got {type(result).__name__}")

            return result

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** (attempt - 1), 20))

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with per-doc text files (e.g. doc_name.txt)")
    ap.add_argument("--model", default="deepseek/deepseek-v3.2", help="Model on OpenRouter")
    ap.add_argument("--api_key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--timeout_s", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=30000)
    ap.add_argument("--no_require_parameters", action="store_true", help="Disable provider.require_parameters")
    args = ap.parse_args()

    # api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    api_key = "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"

    if not api_key:
        raise SystemExit("Missing API key: pass --api_key or set OPENROUTER_API_KEY")

    in_dir = Path(args.in_dir)

    for model_dir in tqdm(list(in_dir.glob("*")), desc="models"):
        if not model_dir.is_dir():
            continue
        
        if model_dir.name not in ['qwen__qwen3-vl-30b-a3b-instruct', 'dpsk-ocr-OCR-1', 'hy_ocr', 'olmocr']:
            continue

        for doc_dir in tqdm(list(model_dir.glob("*")), desc=f"{model_dir.name}/docs", leave=False):
            if not doc_dir.is_dir():
                continue

            doc_md = doc_dir / (doc_dir.name + ".md")
            if doc_md.with_suffix(".json").exists():
                continue
            
            with open(doc_md, "r", encoding="utf-8") as f:
                doc_text = fairy_tame(clean_input_text(f.read()))

            prompt = IE_PROMPT_TEMPLATE.format(doc_text=doc_text)

            parsed = call_openrouter(
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                require_parameters=not args.no_require_parameters,
            )

            raw_json_str = json.dumps(parsed, ensure_ascii=False, indent=2)

            with open(doc_md.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(raw_json_str + "\n")

            with open(doc_md.with_suffix(".json"), "w", encoding="utf-8") as f:
                f.write(raw_json_str + "\n")

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    print("Done.")


if __name__ == "__main__":
    main()
