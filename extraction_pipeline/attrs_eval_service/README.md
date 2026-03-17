# Attrs Evaluation Service (Docker, GPU)

Сервис оценки качества извлечения атрибутов по пайплайну `Attrs_Match_Pipeline.ipynb`.

Вход:
- папка с JSON-файлами стратегий и разметкой `*_gt.json`
- пример структуры: `1_doc.json`, `1_page.json`, `1_table.json`, `1_row.json`, `1_qwen-8b-ft.json`, `1_gt.json`, ...

Выход:
- детальные результаты по каждой паре (TP/FP/FN) в JSON
- сводная таблица в `xlsx`
- сводная таблица в `csv`

## Требования
- Docker Engine
- NVIDIA GPU + `nvidia-container-toolkit`

## Сборка
```bash
cd /Users/yuratomakov/innopolis/pdf2md/results/attrs_eval_service
docker build -t attrs-eval:gpu .
```

## Базовый запуск
```bash
docker run --rm --gpus all \
  -v /abs/path/to/gt_old:/workspace/input \
  -v /abs/path/to/eval_out:/workspace/output \
  attrs-eval:gpu \
  --out_xlsx agg_results.xlsx \
  --match_threshold 0.9
```

## Где будут результаты
- `/workspace/output/*.json` - детальные файлы сравнения для каждой пары документ/стратегия
- `/workspace/output/agg_results.xlsx` - итоговая таблица
- `/workspace/output/agg_results.csv` - итоговая таблица (CSV)

## Полезные опции
- `--debug_subdir gt_old_debug` - писать детальные JSON и таблицы в подпапку
- `--emb_model Qwen/Qwen3-Embedding-4B` - модель эмбеддингов
- `--rerank_model Qwen/Qwen3-Reranker-4B` - модель rerank

Пример с подпапкой:
```bash
docker run --rm --gpus all \
  -v /abs/path/to/gt_old:/workspace/input \
  -v /abs/path/to/eval_out:/workspace/output \
  attrs-eval:gpu \
  --debug_subdir gt_old_debug \
  --out_xlsx agg_results.xlsx \
  --match_threshold 0.9
```

По умолчанию сервис использует:
- `--json_dir /workspace/input`
- `--output_dir /workspace/output`
