#!/usr/bin/env python3
"""PDF-to-Markdown benchmarking pipeline (document-level).

This script evaluates Markdown predictions stored under ./outputs against
Markdown ground-truth files, following the AGENTS.md specification. It
performs the following steps per page:

1. Clean Markdown with dedupe_repetitions.fairy_tame.
2. Normalize basic HTML (<li>, <p>) and strip remaining tags except <img>.
3. Convert Markdown/HTML tables to canonical HTML.
4. Segment pages into headings, paragraphs, list items, tables, and images.
5. Match GT/PRED elements and compute text, table, and image metrics.
6. Emit per-page debug JSON (gt_segmentation.json, pred_segmentation.json,
   match.json) plus per-document and per-model summaries.

Usage:
    python pdf_md_benchmark_doc.py --outputs-root outputs --eval-root eval_out
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup
from Levenshtein import distance as Levenshtein_distance
import numpy as np

from OmniDocBench.utils.data_preprocess import normalized_html_table
from dedupe_repetitions import fairy_tame

try:  # optional embedding deps
    import torch
    from transformers import AutoModel, AutoTokenizer

    HF_EMBED_AVAILABLE = True
except Exception:  # pragma: no cover
    HF_EMBED_AVAILABLE = False
    torch = None  # type: ignore
    AutoModel = AutoTokenizer = None  # type: ignore


TABLE_OVERLAP_THRESHOLD = 0.5
IMAGE_IOU_THRESHOLD = 0.5
TEXT_METEOR_THRESHOLD = 0.2
ROOT_PARENT = "__ROOT__"
CLASS_LABELS = {
    "para": "paragraph",
    "heading": "header",
    "list_item": "list element",
}
DOC_META_PATH = Path("docs_meta.json")


def empty_class_stats() -> Dict[str, Dict[str, int]]:
    return {label: {"tp": 0, "fp": 0, "fn": 0} for label in CLASS_LABELS.values()}


HF_EMBED_MODEL = None
HF_EMBED_TOKENIZER = None
HF_EMBED_DISABLED = False


def load_hf_embedder():
    global HF_EMBED_MODEL, HF_EMBED_TOKENIZER, HF_EMBED_DISABLED
    if HF_EMBED_DISABLED or not HF_EMBED_AVAILABLE:
        return None, None
    if HF_EMBED_MODEL is None or HF_EMBED_TOKENIZER is None:
        try:
            HF_EMBED_TOKENIZER = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
            HF_EMBED_MODEL = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        except Exception:
            HF_EMBED_MODEL = None
            HF_EMBED_TOKENIZER = None
            HF_EMBED_DISABLED = True
    return HF_EMBED_MODEL, HF_EMBED_TOKENIZER


def embed_with_hf(text: str) -> Optional[np.ndarray]:
    text = (text or "").strip()
    if not text:
        return None
    model, tokenizer = load_hf_embedder()
    if model is None or tokenizer is None or torch is None:
        return None
    try:
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings[0].cpu().numpy()
    except Exception:
        return None


def load_doc_meta(path: Path = DOC_META_PATH) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: list(v) for k, v in data.items()}
    except Exception:
        return {}


def tags_for_doc(doc_name: str, meta: Dict[str, List[str]]) -> List[str]:
    candidates = [doc_name]
    if not doc_name.lower().endswith(".pdf"):
        candidates.append(f"{doc_name}.pdf")
    else:
        candidates.append(doc_name[:-4])
    for cand in candidates:
        if cand in meta:
            return meta[cand]
    return []


TABLE_HTML_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
IMG_TAG_RE = re.compile(r"<img\b[^>]*>(?:.*?</img>)?", re.IGNORECASE | re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
ORDERED_LIST_RE = re.compile(r"^\s*(\d+)[\.)]\s+")
BULLET_LIST_RE = re.compile(r"^\s*([-*+])\s+")
SEPARATOR_RE = re.compile(r"^\s*([\-_*~=]{3,})\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SETEXT_HEADING_RE = re.compile(r"^\s*(=+|-+)\s*$")
ANGLE_TOKEN_RE = re.compile(r"<[^<>\n]+>")

DEBUG = bool(os.environ.get("PDF_MD_DEBUG"))


def debug_print(*args) -> None:
    if DEBUG:
        print("[pdf-md-debug]", *args, flush=True)


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return slug.lower() or "doc"


def load_teds_class():
    metrics_path = Path(__file__).resolve().parent / "OmniDocBench" / "metrics" / "table_metric.py"
    spec = importlib.util.spec_from_file_location("pdf_md_benchmark_teds", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TEDS from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.TEDS


TEDS_CLASS = load_teds_class()


def safe_normalized_html(html_text: str) -> str:
    try:
        return normalized_html_table(html_text)
    except Exception:
        return html_text


def list_pages(doc_root: Path) -> Dict[str, Path]:
    if not doc_root.exists():
        return {}
    pages = {path.name: path for path in doc_root.glob("page_*.md")}
    if not pages:
        fallback = doc_root / f"{doc_root.name}.md"
        if fallback.exists():
            pages[fallback.name] = fallback
    return pages


def convert_basic_html_blocks(text: str) -> str:
    def repl_li(match: re.Match) -> str:
        inner = match.group(1).strip()
        return f"\n- {inner}\n"

    def repl_p(match: re.Match) -> str:
        inner = match.group(1).strip()
        return f"\n{inner}\n\n"

    text = LI_RE.sub(repl_li, text)
    text = P_RE.sub(repl_p, text)
    return text


def extract_html_tables(text: str) -> Tuple[str, Dict[str, str]]:
    tables: Dict[str, str] = {}

    def repl(match: re.Match) -> str:
        placeholder = f"[[HTML_TABLE_{len(tables)}]]"
        tables[placeholder] = match.group(0)
        return f"\n{placeholder}\n"

    cleaned = TABLE_HTML_RE.sub(repl, text)
    return cleaned, tables


def strip_html_except_images(text: str) -> str:
    preserved: Dict[str, str] = {}

    def protect(match: re.Match) -> str:
        placeholder = f"[[IMG_TAG_{len(preserved)}]]"
        preserved[placeholder] = match.group(0)
        return placeholder

    text = IMG_TAG_RE.sub(protect, text)
    text = re.sub(r"</?[^>]+>", "", text)
    for placeholder, content in preserved.items():
        text = text.replace(placeholder, content)
    return text


def preprocess_markdown(text: str) -> Tuple[str, Dict[str, str]]:
    text = fairy_tame(text)
    text = text.replace("\r\n", "\n")
    text = convert_basic_html_blocks(text)
    text, html_tables = extract_html_tables(text)
    text = strip_html_except_images(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, html_tables


def split_md_row(line: str) -> List[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in re.split(r"\s*\|\s*", stripped)] if stripped else []


def markdown_table_to_html(lines: Sequence[str]) -> str:
    if len(lines) < 2:
        return ""
    header = split_md_row(lines[0])
    body_lines = [ln for ln in lines[2:] if ln.strip()]
    rows = [split_md_row(ln) for ln in body_lines]
    html_parts = ["<html><body><table border=\"1\">", "<thead><tr>"]
    for cell in header:
        html_parts.append(f"<th>{cell}</th>")
    html_parts.append("</tr></thead><tbody>")
    for row in rows:
        html_parts.append("<tr>")
        for cell in row:
            html_parts.append(f"<td>{cell}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table></body></html>")
    return "".join(html_parts)


def extract_table_cells(html_text: str) -> List[str]:
    if not html_text:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    cells = []
    for cell in soup.find_all(["td", "th"]):
        cells.append(normalize_text(cell.get_text(separator=" ")))
    return cells


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\n", " ")
    text = ANGLE_TOKEN_RE.sub(" ", text)
    text = re.sub(r"[_*`#>\"]", " ", text)
    text = SEPARATOR_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def is_markdown_table_start(lines: Sequence[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    header = lines[idx]
    divider = lines[idx + 1]
    if "|" not in header or "|" not in divider:
        return False
    if not re.search(r"\|-{1,}\|?", divider.replace(" ", "")) and not re.search(r"^-{3,}$", divider.strip("| ")):
        return False
    return True


def collect_markdown_table(lines: Sequence[str], idx: int) -> Tuple[List[str], int]:
    buffer: List[str] = []
    while idx < len(lines) and lines[idx].strip():
        line = lines[idx]
        if "|" not in line:
            break
        buffer.append(line)
        idx += 1
    return buffer, idx


def detect_heading(lines: Sequence[str], idx: int) -> Optional[Tuple[str, int]]:
    line = lines[idx]
    match = HEADING_RE.match(line)
    if match:
        level = len(match.group(1))
        text = match.group(2).strip()
        return text, level
    if idx + 1 < len(lines) and SETEXT_HEADING_RE.match(lines[idx + 1]):
        underline = lines[idx + 1].strip()
        level = 1 if underline.startswith("=") else 2
        return lines[idx].strip(), level
    return None


def strip_list_marker(line: str) -> str:
    if BULLET_LIST_RE.match(line):
        return BULLET_LIST_RE.sub("", line, count=1).strip()
    if ORDERED_LIST_RE.match(line):
        return ORDERED_LIST_RE.sub("", line, count=1).strip()
    return line.strip()


def is_list_item(line: str) -> bool:
    return bool(BULLET_LIST_RE.match(line) or ORDERED_LIST_RE.match(line))


def is_image_line(line: str) -> bool:
    if MARKDOWN_IMAGE_RE.search(line.strip()):
        return True
    return "<img" in line.lower()


def parse_markdown_image(line: str) -> Optional[Dict[str, object]]:
    match = MARKDOWN_IMAGE_RE.search(line)
    if not match:
        return None
    alt_text, src = match.group(1).strip(), match.group(2).strip()
    return {
        "text_raw": line.strip(),
        "text_norm": normalize_text(alt_text),
        "meta": {"src": src, "bbox": None, "desc": alt_text},
    }


def parse_html_image(block: str) -> Optional[Dict[str, object]]:
    tag_match = re.search(r"<img\b([^>]*)>", block, flags=re.IGNORECASE)
    if not tag_match:
        return None
    attr_text = tag_match.group(1)
    attrs = dict(re.findall(r"([a-zA-Z0-9_-]+)=\"([^\"]*)\"", attr_text))
    desc = block[tag_match.end():]
    closing_match = re.search(r"</img>", desc, flags=re.IGNORECASE)
    if closing_match:
        desc = desc[:closing_match.start()]
    else:
        caption_match = re.search(r"</caption>", desc, flags=re.IGNORECASE)
        if caption_match:
            desc = desc[:caption_match.start()]
    desc = desc.strip()
    bbox: Optional[List[float]] = None
    bbox_str = attrs.get("data-bbox")
    if bbox_str:
        tokens = [tok for tok in re.split(r"[\s,]+", bbox_str.strip()) if tok]
        try:
            bbox_vals = [float(val) for val in tokens[:4]]
            if len(bbox_vals) == 4:
                bbox = bbox_vals
        except ValueError:
            bbox = None
    return {
        "text_raw": block.strip(),
        "text_norm": normalize_text(desc),
        "meta": {"bbox": bbox, "attrs": attrs, "desc": desc},
    }


def segment_page(text: str, prefix: str, page_id: str) -> Tuple[Dict[str, object], str]:
    processed, html_tables = preprocess_markdown(text)
    lines = processed.splitlines()
    elements: List[Dict[str, object]] = []
    idx = 0
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter:04d}"

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        if stripped in html_tables:
            raw_html = html_tables[stripped]
            norm_html = safe_normalized_html(raw_html)
            cells = extract_table_cells(norm_html)
            elements.append({
                "id": next_id(),
                "type": "table",
                "text_raw": raw_html,
                "text_norm": norm_html,
                "meta": {"cells": cells, "source": "html"},
            })
            idx += 1
            continue
        heading_info = detect_heading(lines, idx)
        if heading_info:
            text_value, level = heading_info
            if HEADING_RE.match(lines[idx]):
                idx += 1
            else:
                idx += 2
            elements.append({
                "id": next_id(),
                "type": "heading",
                "text_raw": text_value,
                "text_norm": normalize_text(text_value),
                "meta": {"level": level},
            })
            continue
        if is_markdown_table_start(lines, idx):
            table_lines, idx = collect_markdown_table(lines, idx)
            html = markdown_table_to_html(table_lines)
            norm_html = safe_normalized_html(html)
            cells = extract_table_cells(norm_html)
            elements.append({
                "id": next_id(),
                "type": "table",
                "text_raw": "\n".join(table_lines),
                "text_norm": norm_html,
                "meta": {"cells": cells, "source": "markdown"},
            })
            continue
        if is_image_line(stripped):
            md_image = parse_markdown_image(line)
            if md_image:
                md_image.update({"id": next_id(), "type": "image"})
                elements.append(md_image)
                idx += 1
                continue
            # capture until closing tag
            block_lines = [line]
            idx += 1
            if "</img>" not in line.lower():
                while idx < len(lines):
                    block_lines.append(lines[idx])
                    closing_here = "</img>" in lines[idx].lower()
                    idx += 1
                    if closing_here or not block_lines[-1].strip():
                        break
            block = "\n".join(block_lines)
            html_image = parse_html_image(block)
            if html_image:
                html_image.update({"id": next_id(), "type": "image"})
                elements.append(html_image)
                continue
        if is_list_item(line):
            item_lines = [strip_list_marker(line)]
            idx += 1
            while idx < len(lines) and lines[idx].startswith("    "):
                item_lines.append(lines[idx].strip())
                idx += 1
            raw_text = "\n".join(item_lines)
            elements.append({
                "id": next_id(),
                "type": "list_item",
                "text_raw": raw_text,
                "text_norm": normalize_text(raw_text),
                "meta": {},
            })
            continue
        # paragraph fallback
        para_lines = [line]
        idx += 1
        while idx < len(lines):
            peek = lines[idx]
            if not peek.strip():
                break
            if peek.strip() in html_tables:
                break
            if detect_heading(lines, idx) or is_markdown_table_start(lines, idx) or is_image_line(peek.strip()) or is_list_item(peek):
                break
            para_lines.append(peek)
            idx += 1
        raw_para = "\n".join(para_lines).strip()
        if raw_para:
            elements.append({
                "id": next_id(),
                "type": "para",
                "text_raw": raw_para,
                "text_norm": normalize_text(raw_para),
                "meta": {},
            })
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

    return {"page_id": page_id, "elements": elements}, processed


def meteor_score_simplified(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    overlap = sum((Counter(ref_tokens) & Counter(hyp_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    if precision == 0 or recall == 0:
        return 0.0
    alpha = 0.9
    return (precision * recall) / (alpha * precision + (1 - alpha) * recall)


def normalized_edit_distance(a: str, b: str) -> float:
    upper = max(len(a), len(b))
    if upper == 0:
        return 0.0
    return Levenshtein_distance(a, b) / upper


def compute_layout_classification_metrics(class_stats: Dict[str, Dict[str, int]], pair_count: int) -> Dict[str, Any]:
    def safe_mean(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        return statistics.fmean(valid) if valid else None

    per_class: Dict[str, Dict[str, Optional[float]]] = {}
    precision_vals: List[Optional[float]] = []
    recall_vals: List[Optional[float]] = []
    f1_vals: List[Optional[float]] = []
    class_summaries = {}
    for label, counts in class_stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        class_summaries[label] = {"tp": tp, "fp": fp, "fn": fn}

    total_tp = sum(stats["tp"] for stats in class_stats.values())
    total_fp = sum(stats["fp"] for stats in class_stats.values())
    total_fn = sum(stats["fn"] for stats in class_stats.values())
    accuracy = total_tp / pair_count if pair_count else None
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None
    if micro_precision is not None and micro_recall is not None and (micro_precision + micro_recall) > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = None
    return {
        "accuracy": accuracy,
        "macro_precision": safe_mean(precision_vals),
        "macro_recall": safe_mean(recall_vals),
        "macro_f1": safe_mean(f1_vals),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "pair_count": pair_count,
    }


def aggregate_classification_from_items(items: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    totals = empty_class_stats()
    pair_total = 0
    for item in items:
        if not item:
            continue
        per_class = item.get("per_class") or {}
        pair_total += item.get("pair_count") or 0
        for label in CLASS_LABELS.values():
            stats = per_class.get(label, {})
            totals[label]["tp"] += stats.get("tp", 0)
            totals[label]["fp"] += stats.get("fp", 0)
            totals[label]["fn"] += stats.get("fn", 0)
    if pair_total == 0:
        return None
    return compute_layout_classification_metrics(totals, pair_total)


def build_heading_parent_map(elements: List[Dict[str, Any]]) -> Dict[str, str]:
    parents: Dict[str, str] = {}
    stack: List[Tuple[str, int]] = []
    for element in elements:
        if element.get("type") != "heading":
            continue
        level = element.get("meta", {}).get("level")
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 1
        while stack and stack[-1][1] >= level:
            stack.pop()
        parent = stack[-1][0] if stack else ROOT_PARENT
        parents[element["id"]] = parent
        stack.append((element["id"], level))
    return parents


def compute_heading_hierarchy_metrics(gt_elements: List[Dict[str, Any]], pred_elements: List[Dict[str, Any]], heading_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    if not heading_pairs:
        return {
            "parent_accuracy": None,
            "parent_total": 0,
            "parent_correct": 0,
            "edge_precision": None,
            "edge_recall": None,
            "edge_f1": None,
            "edge_tp": 0,
            "edge_fp": 0,
            "edge_fn": 0,
        }
    gt_parents = build_heading_parent_map(gt_elements)
    pred_parents = build_heading_parent_map(pred_elements)
    pred_to_gt = {pred_id: gt_id for gt_id, pred_id in heading_pairs}
    parent_total = 0
    parent_correct = 0
    gt_edges = set()
    pred_edges = set()

    for gt_id, pred_id in heading_pairs:
        parent_total += 1
        gt_parent = gt_parents.get(gt_id, ROOT_PARENT)
        pred_parent = pred_parents.get(pred_id, ROOT_PARENT)
        if pred_parent == ROOT_PARENT:
            mapped_pred_parent = ROOT_PARENT
        else:
            mapped_pred_parent = pred_to_gt.get(pred_parent)
        if mapped_pred_parent is not None and mapped_pred_parent == gt_parent:
            parent_correct += 1
        if gt_parent and gt_parent != ROOT_PARENT:
            gt_edges.add((gt_parent, gt_id))
        if mapped_pred_parent and mapped_pred_parent != ROOT_PARENT:
            pred_edges.add((mapped_pred_parent, gt_id))

    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "parent_accuracy": parent_correct / parent_total if parent_total else None,
        "parent_total": parent_total,
        "parent_correct": parent_correct,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
    }


def aggregate_heading_hierarchy(items: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    parent_totals = 0
    parent_corrects = 0
    edge_tp = edge_fp = edge_fn = 0
    parent_macro_vals: List[Optional[float]] = []
    edge_precision_vals: List[Optional[float]] = []
    edge_recall_vals: List[Optional[float]] = []
    edge_f1_vals: List[Optional[float]] = []
    for item in items:
        if not item:
            continue
        parent_totals += item.get("parent_total", 0)
        parent_corrects += item.get("parent_correct", 0)
        edge_tp += item.get("edge_tp", 0)
        edge_fp += item.get("edge_fp", 0)
        edge_fn += item.get("edge_fn", 0)
        parent_macro_vals.append(item.get("parent_accuracy"))
        edge_precision_vals.append(item.get("edge_precision"))
        edge_recall_vals.append(item.get("edge_recall"))
        edge_f1_vals.append(item.get("edge_f1"))
    if parent_totals == 0 and edge_tp == 0 and edge_fp == 0 and edge_fn == 0 and not parent_macro_vals:
        return None

    def safe_mean(values: List[Optional[float]]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return statistics.fmean(vals) if vals else None

    micro_parent = parent_corrects / parent_totals if parent_totals else None
    micro_precision = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) else None
    micro_recall = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) else None
    if micro_precision is not None and micro_recall is not None and (micro_precision + micro_recall) > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = None

    return {
        "parent_accuracy_macro": safe_mean(parent_macro_vals),
        "parent_accuracy_micro": micro_parent,
        "edge_precision_macro": safe_mean(edge_precision_vals),
        "edge_recall_macro": safe_mean(edge_recall_vals),
        "edge_f1_macro": safe_mean(edge_f1_vals),
        "edge_precision_micro": micro_precision,
        "edge_recall_micro": micro_recall,
        "edge_f1_micro": micro_f1,
        "parent_total": parent_totals,
        "parent_correct": parent_corrects,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
    }


def match_text_elements(gt: List[Dict[str, object]], pred: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    gt_text = [e for e in gt if e["type"] in {"heading", "para", "list_item"}]
    pred_text = [e for e in pred if e["type"] in {"heading", "para", "list_item"}]
    debug_print(f"match_text_elements: {len(gt_text)} GT vs {len(pred_text)} PRED")
    used_pred = set()
    total_chars = 0
    total_edit = 0
    meteor_scores: List[float] = []
    class_stats = empty_class_stats()
    classification_pairs = 0
    heading_pairs: List[Tuple[str, str]] = []

    for idx, g in enumerate(gt_text):
        debug_print(f"  GT {idx}: {g['type']}")
        best = None
        for p in pred_text:
            if p["id"] in used_pred:
                continue
            meteor = meteor_score_simplified(g["text_norm"], p["text_norm"])
            edit = normalized_edit_distance(g["text_norm"], p["text_norm"])
            score = (meteor, -edit)
            if best is None or score > best[0]:
                best = (score, p, meteor, edit)
        if best and best[2] >= TEXT_METEOR_THRESHOLD:
            _, pred_elem, meteor, edit = best
            used_pred.add(pred_elem["id"])
            gt_class = CLASS_LABELS.get(g["type"])
            pred_class = CLASS_LABELS.get(pred_elem["type"])
            if gt_class and pred_class:
                classification_pairs += 1
                if gt_class == pred_class:
                    class_stats[gt_class]["tp"] += 1
                else:
                    class_stats[gt_class]["fn"] += 1
                    class_stats[pred_class]["fp"] += 1
            if g["type"] == "heading" and pred_elem["type"] == "heading":
                heading_pairs.append((g["id"], pred_elem["id"]))
            matches.append({
                "gt_id": g["id"],
                "pred_id": pred_elem["id"],
                "gt_type": g["type"],
                "pred_type": pred_elem["type"],
                "gt_text": g["text_norm"],
                "pred_text": pred_elem["text_norm"],
                "score": {
                    "meteor": meteor,
                    "edit_norm": edit,
                    "meteor_adj": meteor,
                    "edit_adj": edit,
                    "reuse_penalty": 0,
                },
            })
            upper = max(len(g["text_norm"]), len(pred_elem["text_norm"]), 1)
            total_chars += upper
            total_edit += Levenshtein_distance(g["text_norm"], pred_elem["text_norm"])
            meteor_scores.append(meteor)
        else:
            matches.append({
                "gt_id": g["id"],
                "pred_id": None,
                "gt_type": g["type"],
                "pred_type": "none",
                "gt_text": g["text_norm"],
                "pred_text": "",
                "score": {"meteor": 0.0, "edit_norm": 1.0},
            })
            upper = max(len(g["text_norm"]), 1)
            total_chars += upper
            total_edit += upper
            meteor_scores.append(0.0)

    for p in pred_text:
        if p["id"] in used_pred:
            continue
        matches.append({
            "gt_id": None,
            "pred_id": p["id"],
            "gt_type": "none",
            "pred_type": p["type"],
            "gt_text": "",
            "pred_text": p["text_norm"],
            "score": {"meteor": 0.0, "edit_norm": 1.0},
        })
        upper = max(len(p["text_norm"]), 1)
        total_chars += upper
        total_edit += upper

    text_metrics = {
        "edit_norm_len_weighted": (total_edit / total_chars) if total_chars else 0.0,
        "meteor_macro": statistics.fmean(meteor_scores) if meteor_scores else None,
        "char_total": total_chars,
        "edit_total": total_edit,
        "meteor_count": len(meteor_scores),
        "classification": compute_layout_classification_metrics(class_stats, classification_pairs),
        "heading_pairs": heading_pairs,
    }
    return matches, text_metrics


def table_overlap(cells_gt: List[str], cells_pred: List[str]) -> float:
    if not cells_gt:
        return 1.0 if not cells_pred else 0.0
    gt_counter = Counter(cells_gt)
    pred_counter = Counter(cells_pred)
    intersection = sum(min(gt_counter[cell], pred_counter.get(cell, 0)) for cell in gt_counter)
    return intersection / max(len(cells_gt), 1)


def match_tables(gt: List[Dict[str, object]], pred: List[Dict[str, object]], teds_evaluator: Any) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    gt_tables = [e for e in gt if e["type"] == "table"]
    pred_tables = [e for e in pred if e["type"] == "table"]
    matches: List[Dict[str, object]] = []
    used_gt = set()
    used_pred = set()
    overlap_candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_tables):
        for pi, p in enumerate(pred_tables):
            overlap = table_overlap(g["meta"].get("cells", []), p["meta"].get("cells", []))
            if overlap >= TABLE_OVERLAP_THRESHOLD:
                overlap_candidates.append((overlap, gi, pi))
    overlap_candidates.sort(reverse=True)
    teds_scores: List[float] = []
    for overlap, gi, pi in overlap_candidates:
        if gi in used_gt or pi in used_pred:
            continue
        gt_html = g_html = gt_tables[gi]["text_norm"]
        pred_html = pred_tables[pi]["text_norm"]
        try:
            teds_score = teds_evaluator.evaluate(pred_html, g_html)
        except Exception:
            teds_score = 0.0
        teds_scores.append(teds_score)
        matches.append({
            "gt_id": gt_tables[gi]["id"],
            "pred_id": pred_tables[pi]["id"],
            "gt_cells": gt_tables[gi]["meta"].get("cells", []),
            "pred_cells": pred_tables[pi]["meta"].get("cells", []),
            "score": {"cell_overlap": overlap, "teds": teds_score},
        })
        used_gt.add(gi)
        used_pred.add(pi)
    for gi, table in enumerate(gt_tables):
        if gi in used_gt:
            continue
        matches.append({
            "gt_id": table["id"],
            "pred_id": None,
            "gt_cells": table["meta"].get("cells", []),
            "pred_cells": [],
            "score": {"cell_overlap": 0.0, "teds": 0.0},
        })
    for pi, table in enumerate(pred_tables):
        if pi in used_pred:
            continue
        matches.append({
            "gt_id": None,
            "pred_id": table["id"],
            "gt_cells": [],
            "pred_cells": table["meta"].get("cells", []),
            "score": {"cell_overlap": 0.0, "teds": 0.0},
        })

    tp = len(teds_scores)
    fp = len(pred_tables) - tp
    fn = len(gt_tables) - tp
    gt_count = len(gt_tables)
    pred_count = len(pred_tables)
    if gt_count == 0 and pred_count == 0:
        precision = recall = f1 = 1.0
    elif gt_count == 0 and pred_count > 0:
        precision, recall, f1 = 0.0, 1.0, 0.0
    elif gt_count > 0 and pred_count == 0:
        precision, recall, f1 = 1.0, 0.0, 0.0
    else:
        precision = tp / pred_count if pred_count else 1.0
        recall = tp / gt_count if gt_count else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    table_metrics = {
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "teds_mean_on_matched": statistics.fmean(teds_scores) if teds_scores else None,
        "teds_applicable": bool(teds_scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "teds_scores": teds_scores,
    }
    return matches, table_metrics


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max((ax2 - ax1), 0) * max((ay2 - ay1), 0)
    area_b = max((bx2 - bx1), 0) * max((by2 - by1), 0)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def bow_cosine_similarity(text_a: str, text_b: str) -> float:
    tokens_a = normalize_text(text_a).split()
    tokens_b = normalize_text(text_b).split()
    if not tokens_a or not tokens_b:
        return 0.0
    count_a = Counter(tokens_a)
    count_b = Counter(tokens_b)
    intersection = set(count_a) & set(count_b)
    dot = sum(count_a[t] * count_b[t] for t in intersection)
    norm_a = math.sqrt(sum(v * v for v in count_a.values()))
    norm_b = math.sqrt(sum(v * v for v in count_b.values()))
    denom = norm_a * norm_b
    return dot / denom if denom else 0.0


def semantic_text_similarity(text_a: str, text_b: str) -> float:
    emb_a = embed_with_hf(text_a)
    emb_b = embed_with_hf(text_b)
    if emb_a is not None and emb_b is not None:
        denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        if denom:
            return float(np.dot(emb_a, emb_b) / denom)
    return bow_cosine_similarity(text_a, text_b)


def match_images(gt: List[Dict[str, object]], pred: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    gt_images = [e for e in gt if e["type"] == "image"]
    pred_images = [e for e in pred if e["type"] == "image"]
    matches: List[Dict[str, object]] = []
    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_images):
        bbox_g = g.get("meta", {}).get("bbox")
        if not bbox_g:
            continue
        for pi, p in enumerate(pred_images):
            bbox_p = p.get("meta", {}).get("bbox")
            if not bbox_p:
                continue
            iou = bbox_iou(bbox_g, bbox_p)
            if iou >= IMAGE_IOU_THRESHOLD:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    desc_scores: List[float] = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        gt_img = gt_images[gi]
        pred_img = pred_images[pi]
        desc_score = semantic_text_similarity(
            gt_img.get("meta", {}).get("desc", ""),
            pred_img.get("meta", {}).get("desc", ""),
        )
        matches.append({
            "gt_id": gt_img["id"],
            "pred_id": pred_img["id"],
            "gt_bbox": gt_img.get("meta", {}).get("bbox"),
            "pred_bbox": pred_img.get("meta", {}).get("bbox"),
            "score": {"iou": iou, "desc_cosine": desc_score},
            "gt_desc": gt_img.get("meta", {}).get("desc", gt_img.get("text_norm", "")),
            "pred_desc": pred_img.get("meta", {}).get("desc", pred_img.get("text_norm", "")),
        })
        desc_scores.append(desc_score)
        used_gt.add(gi)
        used_pred.add(pi)

    for gi, img in enumerate(gt_images):
        if gi in used_gt:
            continue
        matches.append({
            "gt_id": img["id"],
            "pred_id": None,
            "gt_bbox": img.get("meta", {}).get("bbox"),
            "pred_bbox": None,
            "score": {"iou": 0.0, "desc_cosine": 0.0},
            "gt_desc": img.get("meta", {}).get("desc", img.get("text_norm", "")),
            "pred_desc": "",
        })
    for pi, img in enumerate(pred_images):
        if pi in used_pred:
            continue
        matches.append({
            "gt_id": None,
            "pred_id": img["id"],
            "gt_bbox": None,
            "pred_bbox": img.get("meta", {}).get("bbox"),
            "score": {"iou": 0.0, "desc_cosine": 0.0},
            "gt_desc": "",
            "pred_desc": img.get("meta", {}).get("desc", img.get("text_norm", "")),
        })

    tp = len(used_gt)
    pred_count = len([img for img in pred_images if img.get("meta", {}).get("bbox")])
    gt_count = len([img for img in gt_images if img.get("meta", {}).get("bbox")])
    fp = max(pred_count - tp, 0)
    fn = max(gt_count - tp, 0)

    if gt_count == 0 and pred_count == 0:
        precision = recall = f1 = 1.0
    elif gt_count == 0 and pred_count > 0:
        precision, recall, f1 = 0.0, 1.0, 0.0
    elif gt_count > 0 and pred_count == 0:
        precision, recall, f1 = 1.0, 0.0, 0.0
    else:
        precision = tp / pred_count if pred_count else 1.0
        recall = tp / gt_count if gt_count else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    image_metrics = {
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "desc_cosine_mean_on_matched": statistics.fmean(desc_scores) if desc_scores else None,
        "desc_applicable": bool(desc_scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    return matches, image_metrics


def build_page_metrics(text_metrics: Dict[str, object], table_metrics: Dict[str, object], image_metrics: Dict[str, object], heading_metrics: Dict[str, Any]) -> Dict[str, object]:
    classification = text_metrics.get("classification", {})
    text_classification = {
        "accuracy": classification.get("accuracy"),
        "macro_precision": classification.get("macro_precision"),
        "macro_recall": classification.get("macro_recall"),
        "macro_f1": classification.get("macro_f1"),
        "per_class": classification.get("per_class"),
        "pair_count": classification.get("pair_count"),
    }
    return {
        "text": {
            "edit_norm_len_weighted": text_metrics["edit_norm_len_weighted"],
            "meteor_macro": text_metrics["meteor_macro"],
            "char_total": text_metrics["char_total"],
            "edit_total": text_metrics["edit_total"],
            "meteor_count": text_metrics["meteor_count"],
        },
        "tables": table_metrics,
        "images": image_metrics,
        "text_classification": text_classification,
        "heading_hierarchy": heading_metrics,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content or "")


def aggregate_scalar(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.fmean(vals)


def aggregate_micro(tp: int, fp: int, fn: int) -> Dict[str, float]:
    pred_count = tp + fp
    gt_count = tp + fn
    if gt_count == 0 and pred_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if gt_count == 0 and pred_count > 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if gt_count > 0 and pred_count == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    precision = tp / pred_count if pred_count else 1.0
    recall = tp / gt_count if gt_count else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def process_page(gt_path: Optional[Path], pred_path: Optional[Path], page_id: str, out_dir: Path, teds_evaluator: Any) -> Dict[str, object]:
    gt_text = gt_path.read_text(encoding="utf-8") if gt_path and gt_path.exists() else ""
    pred_text = pred_path.read_text(encoding="utf-8") if pred_path and pred_path.exists() else ""

    gt_segments, gt_processed = segment_page(gt_text, "gt", page_id)
    pred_segments, pred_processed = segment_page(pred_text, "pred", page_id)

    text_matches, text_metrics = match_text_elements(gt_segments["elements"], pred_segments["elements"])
    table_matches, table_metrics = match_tables(gt_segments["elements"], pred_segments["elements"], teds_evaluator)
    image_matches, image_metrics = match_images(gt_segments["elements"], pred_segments["elements"])

    heading_metrics = compute_heading_hierarchy_metrics(
        gt_segments["elements"],
        pred_segments["elements"],
        text_metrics.get("heading_pairs", []),
    )

    page_metrics = build_page_metrics(text_metrics, table_metrics, image_metrics, heading_metrics)

    ensure_dir(out_dir)
    write_json(out_dir / "gt_segmentation.json", gt_segments)
    write_json(out_dir / "pred_segmentation.json", pred_segments)
    write_text(out_dir / "gt_processed.md", gt_processed)
    write_text(out_dir / "pred_processed.md", pred_processed)
    write_json(out_dir / "match.json", {
        "page_id": page_id,
        "text_matches": text_matches,
        "table_matches": table_matches,
        "image_matches": image_matches,
        "page_metrics": page_metrics,
    })

    return {
        "page_id": page_id,
        "metrics": page_metrics,
    }


def aggregate_document(doc_name: str, doc_slug: str, pages: List[Dict[str, object]]) -> Dict[str, object]:
    text_macro = aggregate_scalar(p["metrics"]["text"]["edit_norm_len_weighted"] for p in pages)
    meteor_macro = aggregate_scalar(p["metrics"]["text"]["meteor_macro"] for p in pages)
    text_char_total = sum(p["metrics"]["text"]["char_total"] for p in pages)
    text_edit_total = sum(p["metrics"]["text"]["edit_total"] for p in pages)
    text_micro = (text_edit_total / text_char_total) if text_char_total else None

    table_macro = {
        "precision": aggregate_scalar(p["metrics"]["tables"]["det_precision"] for p in pages),
        "recall": aggregate_scalar(p["metrics"]["tables"]["det_recall"] for p in pages),
        "f1": aggregate_scalar(p["metrics"]["tables"]["det_f1"] for p in pages),
    }
    table_tp = sum(p["metrics"]["tables"].get("tp", 0) for p in pages)
    table_fp = sum(p["metrics"]["tables"].get("fp", 0) for p in pages)
    table_fn = sum(p["metrics"]["tables"].get("fn", 0) for p in pages)
    table_micro = aggregate_micro(table_tp, table_fp, table_fn)
    table_teds = aggregate_scalar(p["metrics"]["tables"].get("teds_mean_on_matched") for p in pages if p["metrics"]["tables"].get("teds_applicable"))
    teds_applicability = sum(1 for p in pages if p["metrics"]["tables"].get("teds_applicable")) / len(pages) if pages else 0.0

    image_macro = {
        "precision": aggregate_scalar(p["metrics"]["images"]["det_precision"] for p in pages),
        "recall": aggregate_scalar(p["metrics"]["images"]["det_recall"] for p in pages),
        "f1": aggregate_scalar(p["metrics"]["images"]["det_f1"] for p in pages),
    }
    image_tp = sum(p["metrics"]["images"].get("tp", 0) for p in pages)
    image_fp = sum(p["metrics"]["images"].get("fp", 0) for p in pages)
    image_fn = sum(p["metrics"]["images"].get("fn", 0) for p in pages)
    image_micro = aggregate_micro(image_tp, image_fp, image_fn)
    image_cos = aggregate_scalar(p["metrics"]["images"].get("desc_cosine_mean_on_matched") for p in pages if p["metrics"]["images"].get("desc_applicable"))
    cosine_app = sum(1 for p in pages if p["metrics"]["images"].get("desc_applicable")) / len(pages) if pages else 0.0

    classification_metrics = aggregate_classification_from_items(
        (p["metrics"].get("text_classification") for p in pages)
    )
    heading_metrics = aggregate_heading_hierarchy(
        (p["metrics"].get("heading_hierarchy") for p in pages)
    )

    return {
        "doc_name": doc_name,
        "doc_id": doc_slug,
        "page_count": len(pages),
        "metrics": {
            "text": {
                "edit_norm_macro": text_macro,
                "edit_norm_micro": text_micro,
                "meteor_macro": meteor_macro,
                "classification_macro_f1": classification_metrics.get("macro_f1") if classification_metrics else None,
                "classification_micro_f1": classification_metrics.get("micro_f1") if classification_metrics else None,
            },
            "tables": {
                "macro": table_macro,
                "micro": table_micro,
                "teds_mean_on_matched": table_teds,
                "teds_applicability": teds_applicability,
            },
            "images": {
                "macro": image_macro,
                "micro": image_micro,
                "desc_cosine_mean_on_matched": image_cos,
                "desc_applicability": cosine_app,
            },
            "text_classification": classification_metrics,
            "heading_hierarchy": heading_metrics,
        },
    }


def aggregate_dataset(documents: List[Dict[str, object]]) -> Dict[str, object]:
    text_macro = aggregate_scalar(doc["metrics"]["text"]["edit_norm_macro"] for doc in documents)
    text_micro = aggregate_scalar(doc["metrics"]["text"]["edit_norm_micro"] for doc in documents)
    meteor_macro = aggregate_scalar(doc["metrics"]["text"]["meteor_macro"] for doc in documents)

    def aggregate_family(documents: List[Dict[str, object]], family: str) -> Dict[str, object]:
        macro_precision = aggregate_scalar(doc["metrics"][family]["macro"]["precision"] for doc in documents)
        macro_recall = aggregate_scalar(doc["metrics"][family]["macro"]["recall"] for doc in documents)
        macro_f1 = aggregate_scalar(doc["metrics"][family]["macro"]["f1"] for doc in documents)
        micro_precision = aggregate_scalar(doc["metrics"][family]["micro"]["precision"] for doc in documents)
        micro_recall = aggregate_scalar(doc["metrics"][family]["micro"]["recall"] for doc in documents)
        micro_f1 = aggregate_scalar(doc["metrics"][family]["micro"]["f1"] for doc in documents)
        extra_keys = {}
        if family == "tables":
            extra_keys = {
                "teds_mean_on_matched": aggregate_scalar(doc["metrics"][family]["teds_mean_on_matched"] for doc in documents),
                "teds_applicability": aggregate_scalar(doc["metrics"][family]["teds_applicability"] for doc in documents),
            }
        if family == "images":
            extra_keys = {
                "desc_cosine_mean_on_matched": aggregate_scalar(doc["metrics"][family]["desc_cosine_mean_on_matched"] for doc in documents),
                "desc_applicability": aggregate_scalar(doc["metrics"][family]["desc_applicability"] for doc in documents),
            }
        return {
            "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
            "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
            **extra_keys,
        }

    classification = aggregate_classification_from_items(
        (doc["metrics"].get("text_classification") for doc in documents)
    )
    heading_metrics = aggregate_heading_hierarchy(
        (doc["metrics"].get("heading_hierarchy") for doc in documents)
    )

    return {
        "text": {
            "edit_norm_macro": text_macro,
            "edit_norm_micro": text_micro,
            "meteor_macro": meteor_macro,
            "classification_macro_f1": classification.get("macro_f1") if classification else None,
            "classification_micro_f1": classification.get("micro_f1") if classification else None,
        },
        "tables": aggregate_family(documents, "tables"),
        "images": aggregate_family(documents, "images"),
        "classification": classification,
        "heading_hierarchy": heading_metrics,
    }


def flatten_dataset_metrics(model_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    text = metrics.get("text", {})
    tables = metrics.get("tables", {})
    images = metrics.get("images", {})
    classification = metrics.get("classification", {}) or {}
    heading = metrics.get("heading_hierarchy", {}) or {}
    per_class = classification.get("per_class", {}) if classification else {}
    table_macro = tables.get("macro", {})
    table_micro = tables.get("micro", {})
    image_macro = images.get("macro", {})
    image_micro = images.get("micro", {})
    return {
        "model_id": model_id,
        "text_edit_norm_macro": text.get("edit_norm_macro"),
        "text_edit_norm_micro": text.get("edit_norm_micro"),
        "text_meteor_macro": text.get("meteor_macro"),
        "text_classification_macro_f1": text.get("classification_macro_f1"),
        "text_classification_micro_f1": text.get("classification_micro_f1"),
        "table_macro_precision": table_macro.get("precision"),
        "table_macro_recall": table_macro.get("recall"),
        "table_macro_f1": table_macro.get("f1"),
        "table_micro_precision": table_micro.get("precision"),
        "table_micro_recall": table_micro.get("recall"),
        "table_micro_f1": table_micro.get("f1"),
        "table_teds_mean_on_matched": tables.get("teds_mean_on_matched"),
        "table_teds_applicability": tables.get("teds_applicability"),
        "image_macro_precision": image_macro.get("precision"),
        "image_macro_recall": image_macro.get("recall"),
        "image_macro_f1": image_macro.get("f1"),
        "image_micro_precision": image_micro.get("precision"),
        "image_micro_recall": image_micro.get("recall"),
        "image_micro_f1": image_micro.get("f1"),
        "image_desc_cosine_mean_on_matched": images.get("desc_cosine_mean_on_matched"),
        "image_desc_applicability": images.get("desc_applicability"),
        "lists_f1": (per_class.get("list element") or {}).get("f1"),
        "headers_f1": (per_class.get("header") or {}).get("f1"),
        "paras_f1": (per_class.get("paragraph") or {}).get("f1"),
        "heading_parent_acc_macro": heading.get("parent_accuracy_macro"),
        "heading_parent_acc_micro": heading.get("parent_accuracy_micro"),
        "heading_edge_f1_macro": heading.get("edge_f1_macro"),
        "heading_edge_f1_micro": heading.get("edge_f1_micro"),
    }


def write_bench_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "model_id",
        "text_edit_norm_macro",
        "text_edit_norm_micro",
        "text_meteor_macro",
        "text_classification_macro_f1",
        "text_classification_micro_f1",
        "table_macro_precision",
        "table_macro_recall",
        "table_macro_f1",
        "table_micro_precision",
        "table_micro_recall",
        "table_micro_f1",
        "table_teds_mean_on_matched",
        "table_teds_applicability",
        "image_macro_precision",
        "image_macro_recall",
        "image_macro_f1",
        "image_micro_precision",
        "image_micro_recall",
        "image_micro_f1",
        "image_desc_cosine_mean_on_matched",
        "image_desc_applicability",
        "lists_f1",
        "headers_f1",
        "paras_f1",
        "heading_parent_acc_macro",
        "heading_parent_acc_micro",
        "heading_edge_f1_macro",
        "heading_edge_f1_micro",
    ]
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            sanitized = {key: ("" if row.get(key) is None else row.get(key)) for key in headers}
            writer.writerow(sanitized)


def flatten_document_metrics(model_id: str, doc_summary: Dict[str, Any], doc_tags: List[str]) -> Dict[str, Any]:
    metrics = doc_summary.get("metrics", {})
    text = metrics.get("text", {})
    tables = metrics.get("tables", {})
    images = metrics.get("images", {})
    classification = metrics.get("text_classification", {}) or {}
    heading = metrics.get("heading_hierarchy", {}) or {}
    per_class = classification.get("per_class", {}) if classification else {}
    table_micro = tables.get("micro", {})
    image_micro = images.get("micro", {})
    text_edit_micro = text.get("edit_norm_micro")
    text_meteor_micro = text.get("meteor_macro")
    classification_macro = text.get("classification_macro_f1")
    table_micro_f1 = table_micro.get("f1")
    table_teds = tables.get("teds_mean_on_matched")
    heading_edge_micro = heading.get("edge_f1_micro")
    image_micro_f1 = image_micro.get("f1")
    image_desc_cosine = images.get("desc_cosine_mean_on_matched")

    e_prime = 1 - text_edit_micro if text_edit_micro is not None else 0.0
    m_component = text_meteor_micro if text_meteor_micro is not None else 0.0
    c_component = classification_macro if classification_macro is not None else 0.0
    t_component = (table_micro_f1 if table_micro_f1 is not None else 0.0) * (
        table_teds if table_teds is not None else 0.0
    )
    h_component = heading_edge_micro if heading_edge_micro is not None else 0.0
    image_micro_f1_val = image_micro_f1 if image_micro_f1 is not None else 0.0
    image_desc_cosine_val = image_desc_cosine if image_desc_cosine is not None else 0.0
    i_component = (3 * image_micro_f1_val + image_desc_cosine_val) / 4
    score = (
        0.25 * e_prime
        + 0.25 * m_component
        + 0.15 * c_component
        + 0.20 * t_component
        + 0.10 * h_component
        + 0.05 * i_component
    )
    return {
        "model_id": model_id,
        "pdf_name": doc_summary.get("doc_name"),
        "scan": int("scan" in doc_tags),
        "table": int("table" in doc_tags),
        "rotation": int("rotation" in doc_tags),
        "draw": int("draw" in doc_tags),
        "text_edit_norm_micro": text_edit_micro,
        "text_meteor_micro": text_meteor_micro,
        "text_classification_macro_f1": text.get("classification_macro_f1"),
        "table_micro_precision": table_micro.get("precision"),
        "table_micro_recall": table_micro.get("recall"),
        "table_micro_f1": table_micro_f1,
        "table_teds_mean_on_matched": table_teds,
        "image_micro_precision": image_micro.get("precision"),
        "image_micro_recall": image_micro.get("recall"),
        "image_micro_f1": image_micro_f1,
        "image_desc_cosine_mean_on_matched": image_desc_cosine,
        "image_desc_applicability": images.get("desc_applicability"),
        "lists_f1": (per_class.get("list element") or {}).get("f1"),
        "headers_f1": (per_class.get("header") or {}).get("f1"),
        "paras_f1": (per_class.get("paragraph") or {}).get("f1"),
        "heading_edge_f1_micro": heading.get("edge_f1_micro"),
        "text_accuracy_component": e_prime,
        "text_semantics_component": m_component,
        "text_structure_component": c_component,
        "tables_component": t_component,
        "hierarchy_component": h_component,
        "images_component": i_component,
        "score": score,
    }


def write_doc_bench_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "model_id",
        "pdf_name",
        "scan",
        "table",
        "rotation",
        "draw",
        "text_edit_norm_micro",
        "text_meteor_micro",
        "text_classification_macro_f1",
        "table_micro_precision",
        "table_micro_recall",
        "table_micro_f1",
        "table_teds_mean_on_matched",
        "image_micro_precision",
        "image_micro_recall",
        "image_micro_f1",
        "image_desc_cosine_mean_on_matched",
        "image_desc_applicability",
        "lists_f1",
        "headers_f1",
        "paras_f1",
        "heading_edge_f1_micro",
        "text_accuracy_component",
        "text_semantics_component",
        "text_structure_component",
        "tables_component",
        "hierarchy_component",
        "images_component",
        "score",
    ]
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            sanitized = {key: ("" if row.get(key) is None else row.get(key)) for key in headers}
            writer.writerow(sanitized)


def process_document(doc_path: Path, pred_doc_path: Path, model_out_root: Path, teds_evaluator: Any) -> Dict[str, object]:
    doc_name = doc_path.name
    doc_slug = slugify(doc_name)
    doc_out_root = model_out_root / doc_slug
    page_id = f"{doc_slug}_doc"
    gt_page = doc_path / f"{doc_path.name}.md"
    if not gt_page.exists():
        gt_candidates = sorted(doc_path.glob("*.md"))
        gt_page = gt_candidates[0] if gt_candidates else None
    pred_page = pred_doc_path / gt_page.name if gt_page and (pred_doc_path / gt_page.name).exists() else None
    page_out_dir = doc_out_root / "doc"
    print(f"    [pdf-md-doc] Document-level evaluation for {doc_name}")
    record = process_page(gt_page, pred_page, page_id, page_out_dir, teds_evaluator)
    return aggregate_document(doc_name, doc_slug, [record])


def run_evaluation(outputs_root: Path, eval_root: Path) -> None:
    gt_root = outputs_root / "ground_truth"
    if not gt_root.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_root}")
    prediction_dirs = [d for d in outputs_root.iterdir() if d.is_dir() and d.name != "ground_truth"]
    print(f"[pdf-md] Found {len(prediction_dirs)} model directories under {outputs_root}")
    teds = TEDS_CLASS(structure_only=False)
    doc_meta = load_doc_meta()

    bench_rows: List[Dict[str, Any]] = []
    doc_rows: List[Dict[str, Any]] = []
    for model_dir in prediction_dirs:
        model_slug = slugify(model_dir.name)
        model_out_root = eval_root / model_slug
        print(f"[pdf-md] Processing model: {model_dir.name} -> {model_slug}")
        documents: List[Dict[str, object]] = []
        for doc_path in sorted(gt_root.iterdir()):
            if not doc_path.is_dir():
                continue
            pred_doc_path = model_dir / doc_path.name
            print(f"  [pdf-md] Document: {doc_path.name}")
            doc_summary = process_document(doc_path, pred_doc_path, model_out_root, teds)
            documents.append(doc_summary)
            tags = tags_for_doc(doc_path.name, doc_meta)
            doc_rows.append(flatten_document_metrics(model_dir.name, doc_summary, tags))
        summary = {"model_id": model_dir.name, "documents": documents}
        dataset_summary = aggregate_dataset(documents)
        write_json(model_out_root / "summary_document.json", summary)
        write_json(model_out_root / "summary_dataset.json", {
            "model_id": model_dir.name,
            "metrics": dataset_summary,
        })
        bench_rows.append(flatten_dataset_metrics(model_dir.name, dataset_summary))

    write_bench_csv(eval_root / "bench.csv", bench_rows)
    write_doc_bench_csv(eval_root / "bench_docs.csv", doc_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Markdown predictions against ground truth")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Directory with ground_truth and model outputs")
    parser.add_argument("--eval-root", type=Path, default=Path("eval_out"), help="Directory to write evaluation artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args.outputs_root, args.eval_root)


if __name__ == "__main__":
    main()
