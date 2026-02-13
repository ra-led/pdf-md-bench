# NVIDIA GPU Container Deploy

## Prerequisites
- Docker Engine with NVIDIA runtime support (`nvidia-container-toolkit`).
- Host GPU drivers installed and visible with `nvidia-smi`.
- OpenRouter API key exported on host:

```bash
export OPENROUTER_API_KEY='your_key_here'
```

## Build image

```bash
docker build -t pdf2char:gpu -f /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/Dockerfile /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline
```

## Run one document

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/input:/workspace/input \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/output:/workspace/output \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.json \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate
```

## Run tables-only pipeline

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -e PIPELINE_SCRIPT=pdf_to_characteristics_json_tables.py \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/input:/workspace/input \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/output:/workspace/output \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.tables.json \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate \
  --save_tables_dir /workspace/work/tables
```

## Run pages-only pipeline

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -e PIPELINE_SCRIPT=pdf_to_characteristics_json_pages.py \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/input:/workspace/input \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/output:/workspace/output \
  -v /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline/work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.pages.json \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate \
  --save_pages_dir /workspace/work/pages
```

## Run with compose

```bash
cd /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline
docker compose -f docker-compose.gpu.yml run --rm pdf2char \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.json \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate
```

## Notes
- First build is large and may take significant time because CUDA + torch + vLLM + model deps are installed.
- Model weights are pulled at runtime by the DeepSeek OCR runner from Hugging Face.
- If needed, pass extra extraction args directly (for example `--openrouter_model ...`, `--max_tokens ...`).
- For tables-only run, extraction is performed table-by-table and neighboring table boundaries are checked/merged with an additional LLM confirmation step.
- For pages-only run, extraction is performed page-by-page and neighboring page boundaries are checked/merged with an additional LLM confirmation step.
