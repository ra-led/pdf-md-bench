# PDF -> Attrs (Qwen FT, Docker)

Сервис для извлечения атрибутов напрямую из PDF по пайплайну из `Visual_IE_Inference.ipynb`.

## Требования
- Docker Engine
- NVIDIA GPU + `nvidia-container-toolkit`
- Смонтированная папка с PDF
- Смонтированная папка с чекпоинтом адаптера (`checkpoint-*`)

## Сборка
```bash
cd ./pdf_visual_ie_service
docker build -t pdf-visual-ie:gpu .
```

## Запуск
```bash
docker run --rm --gpus all \
  -v /abs/path/to/pdfs:/workspace/input \
  -v /abs/path/to/output:/workspace/output \
  -v /abs/path/to/adapter:/workspace/adapter \
  -v /abs/path/to/work:/workspace/work \
  pdf-visual-ie:gpu
```

Например:
```bash
docker run --rm --gpus all \
  -v /home/ubuntu/pdf2md_service/pdfs:/workspace/input \
  -v ./output:/workspace/output \
  -v ./checkpoint-50:/workspace/adapter \
  -v ./work:/workspace/work \
  pdf-visual-ie:gpu \
  --num_beams 1
  --vllm_max_model_len 6000
```

## Параметры
- `--input_pdf_dir`: по умолчанию `/workspace/input`
- `--output_dir`: по умолчанию `/workspace/output`
- `--adapter_checkpoint`: по умолчанию `/workspace/adapter/checkpoint-50`
- `--work_dir`: по умолчанию `/workspace/work`
- `--base_model`: по умолчанию `Qwen/Qwen3-VL-8B-Instruct`
- `--temperature`, `--num_beams`, `--max_new_tokens`
- `--vllm_max_model_len`, `--vllm_gpu_memory_utilization`
- `--scale`: масштаб рендера PDF-страниц

## Выходные файлы
В `output_dir`:
- `<pdf_stem>_qwen-8b-ft.json` для каждого PDF из входной папки.
