#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def attr_to_text(attr: dict) -> str:
    name = str(attr.get("name", "")).strip()
    value = str(attr.get("request_value", "")).strip()
    return f"{name}: {value}"


def format_instruction(instruction: str, query: str, doc: str):
    return [
        {
            "role": "system",
            "content": (
                'Judge whether the Document meets the requirements based on the Query '
                'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            ),
        },
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
        },
    ]


def process_rerank_pair(
    tokenizer,
    query_text: str,
    doc_text: str,
    instruction: str,
    max_length: int,
    suffix_tokens: list[int],
):
    messages = [format_instruction(instruction, query_text, doc_text)]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    tokenized = tokenized[0][:max_length] + suffix_tokens
    return TokensPrompt(prompt_token_ids=tokenized)


def compute_yes_probability(
    rerank_model,
    prompt,
    sampling_params,
    true_token: int,
    false_token: int,
) -> float:
    outputs = rerank_model.generate([prompt], sampling_params, use_tqdm=False)
    final_logits = outputs[0].outputs[0].logprobs[-1]

    true_logit = final_logits[true_token].logprob if true_token in final_logits else -10
    false_logit = final_logits[false_token].logprob if false_token in final_logits else -10

    true_score = math.exp(true_logit)
    false_score = math.exp(false_logit)
    return true_score / (true_score + false_score)


def evaluate_attrs_match(
    attrs_gt: list[dict],
    attrs_pred: list[dict],
    emb_model,
    rerank_model,
    tokenizer,
    emb_task: str,
    rer_task: str,
    match_threshold: float = 0.8,
    max_length: int = 8192,
):
    if not attrs_gt and not attrs_pred:
        return []

    gt_texts_plain = [attr_to_text(a) for a in attrs_gt]
    pred_texts_plain = [attr_to_text(a) for a in attrs_pred]

    gt_texts_embed = [get_detailed_instruct(emb_task, t) for t in gt_texts_plain]
    pred_texts_embed = pred_texts_plain

    emb_inputs = gt_texts_embed + pred_texts_embed
    emb_outputs = emb_model.embed(emb_inputs)
    embeddings = torch.tensor([o.outputs.embedding for o in emb_outputs], dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    gt_emb = embeddings[: len(attrs_gt)]
    pred_emb = embeddings[len(attrs_gt) :]
    sim_matrix = gt_emb @ pred_emb.T

    candidate_pairs = []
    for i in range(len(attrs_gt)):
        for j in range(len(attrs_pred)):
            candidate_pairs.append((i, j, float(sim_matrix[i, j])))
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    matched_gt = set()
    matched_pred = set()
    results = []

    for i, j, sim_score in candidate_pairs:
        if i in matched_gt or j in matched_pred:
            continue

        query_text = gt_texts_plain[i]
        doc_text = pred_texts_plain[j]

        prompt = process_rerank_pair(
            tokenizer=tokenizer,
            query_text=query_text,
            doc_text=doc_text,
            instruction=rer_task,
            max_length=max_length,
            suffix_tokens=suffix_tokens,
        )

        final_score = compute_yes_probability(
            rerank_model=rerank_model,
            prompt=prompt,
            sampling_params=sampling_params,
            true_token=true_token,
            false_token=false_token,
        )

        if final_score > match_threshold:
            matched_gt.add(i)
            matched_pred.add(j)
            results.append(
                {
                    "gt_attr": attrs_gt[i],
                    "pred_attr": attrs_pred[j],
                    "result": "TP",
                    "sim": sim_score,
                    "final_score": final_score,
                }
            )

    for j, pred_attr in enumerate(attrs_pred):
        if j not in matched_pred:
            results.append(
                {
                    "gt_attr": None,
                    "pred_attr": pred_attr,
                    "result": "FP",
                    "sim": None,
                    "final_score": None,
                }
            )

    for i, gt_attr in enumerate(attrs_gt):
        if i not in matched_gt:
            results.append(
                {
                    "gt_attr": gt_attr,
                    "pred_attr": None,
                    "result": "FN",
                    "sim": None,
                    "final_score": None,
                }
            )

    return results


def compute_prf(results):
    tp = sum(1 for r in results if r["result"] == "TP")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_attrs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_folder(
    json_dir: Path,
    emb_model,
    rerank_model,
    tokenizer,
    emb_task: str,
    rer_task: str,
    out_xlsx: str,
    match_threshold: float,
    debug_dir: Path,
):
    debug_dir.mkdir(parents=True, exist_ok=True)

    gt_files = {}
    pred_files = []

    for p in json_dir.glob("*.json"):
        stem = p.stem
        if "_" not in stem:
            continue
        doc_id, strategy = stem.split("_", 1)
        if strategy == "gt":
            gt_files[doc_id] = p
        else:
            pred_files.append((doc_id, strategy, p))

    rows = []
    for doc_id, strategy, pred_path in tqdm(sorted(pred_files)):
        if doc_id not in gt_files:
            print(f"GT missing for {pred_path.name}")
            continue

        gt_path = gt_files[doc_id]
        attrs_gt = load_attrs(gt_path)
        attrs_pred = load_attrs(pred_path)

        results = evaluate_attrs_match(
            attrs_gt=attrs_gt,
            attrs_pred=attrs_pred,
            emb_model=emb_model,
            rerank_model=rerank_model,
            tokenizer=tokenizer,
            match_threshold=match_threshold,
            emb_task=emb_task,
            rer_task=rer_task,
        )

        (debug_dir / pred_path.name).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

        metrics = compute_prf(results)
        rows.append(
            {
                "DOC_NAME": int(doc_id) if doc_id.isdigit() else doc_id,
                "STRATEGY": strategy,
                "PAIR_F1": metrics["f1"],
                "PAIR_REC": metrics["recall"],
                "PAIR_PRC": metrics["precision"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["DOC_NAME", "STRATEGY", "PAIR_F1", "PAIR_REC", "PAIR_PRC", "TP", "FP", "FN"])
    else:
        df = df.sort_values(["DOC_NAME", "STRATEGY"]).reset_index(drop=True)

    csv_name = out_xlsx.replace(".xlsx", ".csv") if out_xlsx.endswith(".xlsx") else f"{out_xlsx}.csv"
    df[["DOC_NAME", "STRATEGY", "PAIR_F1", "PAIR_REC", "PAIR_PRC"]].to_csv(debug_dir / csv_name, index=False)

    with pd.ExcelWriter(debug_dir / out_xlsx) as writer:
        df[["DOC_NAME", "STRATEGY", "PAIR_F1", "PAIR_REC", "PAIR_PRC"]].to_excel(
            writer, index=False, sheet_name="eval_table_docs_strategies"
        )
        if not df.empty:
            (
                df.groupby("STRATEGY")[["PAIR_F1", "PAIR_REC", "PAIR_PRC"]]
                .mean()
                .sort_values("PAIR_F1", ascending=False)
                .to_excel(writer, sheet_name="strategy_avg")
            )
        else:
            pd.DataFrame(columns=["PAIR_F1", "PAIR_REC", "PAIR_PRC"]).to_excel(
                writer, sheet_name="strategy_avg"
            )

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate extracted attrs against *_gt.json labels.")
    parser.add_argument("--json_dir", required=True, help="Directory with *.json (includes *_gt.json and predictions).")
    parser.add_argument("--output_dir", required=True, help="Directory for detailed pair json + csv/xlsx summary.")
    parser.add_argument("--out_xlsx", default="agg_results.xlsx", help="Summary xlsx file name.")
    parser.add_argument("--debug_subdir", default="", help="Optional subdir inside output_dir for per-pair details.")
    parser.add_argument("--match_threshold", type=float, default=0.9, help="Threshold for TP after rerank score.")
    parser.add_argument("--emb_model", default="Qwen/Qwen3-Embedding-4B", help="Embedding model for vLLM.")
    parser.add_argument("--rerank_model", default="Qwen/Qwen3-Reranker-4B", help="Reranker model for vLLM.")
    return parser.parse_args()


def main():
    args = parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / args.debug_subdir if args.debug_subdir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not json_dir.exists():
        raise FileNotFoundError(f"json_dir not found: {json_dir}")

    print("[eval] loading models...")
    emb_model = LLM(model=args.emb_model, task="embed")
    rerank_model = LLM(
        model=args.rerank_model,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_model_len=5000,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.rerank_model)

    rer_task = (
        "Decide whether two requirement/characteristic name and values refer to the SAME requirement.\n"
        '- "yes" if Query and Document are paraphrases / same meaning / same field, even with minor formatting or punctuation changes.\n'
        '- "no" if meaning differs, scope differs, or one is only a partial subset of the other (missing essential clauses), diffirent in wording small but significant.'
    )
    emb_task = (
        "Given a characteristic search query, retrieve relevant characteristic that pure matches "
        "name and value of the query characteristic"
    )

    df = evaluate_folder(
        json_dir=json_dir,
        emb_model=emb_model,
        rerank_model=rerank_model,
        tokenizer=tokenizer,
        rer_task=rer_task,
        emb_task=emb_task,
        out_xlsx=args.out_xlsx,
        match_threshold=args.match_threshold,
        debug_dir=debug_dir,
    )

    print("[eval] done")
    print(df)


if __name__ == "__main__":
    main()
