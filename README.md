# pdf-md-bench

Набор ноутбуков и утилит для сравнения PDF → Markdown конвейеров. Репозиторий содержит всё, что нужно, чтобы прогнать документы через разные OCR/LLM-модели, привести результаты к единому формату и посчитать метрики качества (на базе `OmniDocBench`).

## Ход исследования

### 1. Предварительный отбор моделей (качественный smoke-тест)

*`PDF_MD_Smoke_Test_1.ipynb`* — первый набор ручных проверок. В ноутбуке каждая секция посвящена отдельному инструменту, а страницы PDF сначала приводятся к PNG, после чего вызываются соответствующие API/SDK. По заголовкам можно восстановить полный список протестированных решений:

- MinerU
- MinerU-VLM
- Docling
- Docling VLM
- PP Structure V3
- Monkey-OCR
- Nougat
- Dots.OCR
- DeepSeek-OCR
- Dolphin (OmniParse)

*`PDF_MD_Smoke_Test_2.ipynb`* — второй набор проверок:

- **HF**: локальный запуск `baidu/ERNIE-4.5-VL-28B-A3B-PT` через `transformers` с подготовкой jsonl-датасета (страницы рендерятся через `pypdfium2`).
- **MS-SWIFT**: генерация команд `swift infer` для `Qwen/Qwen3-VL-30B-A3B-Instruct`, `zai-org/GLM-4.6V-Flash`, `OpenGVLab/InternVL3-14B-Instruct`, `mistralai/Ministral-3-14B-Instruct-2512`, `LLM-Research/gemma-3-27b-it` с двумя вариантами бэкенда (`vllm` и чистый PyTorch).

*`pdf_to_md_openrouter.py`* — скрипт для прогона страниц PDF через OpenRouter.

### 2. Конвертация с оптимальными параметрами (количественный прогон)

*`pdf_to_md_openrouter_bench.py`* скрипт для прогона страниц PDF через OpenRouter c оптимальными параметрами и промтом для финального теста без постпроцессинга изображений.

*`Bench_OlmOCR.ipynb`*, *`Bench_Hunyuan_OCR.ipynb`* и *`Bench_DeepSeek_OCR.ipynb`* конвертация в md для финального теста с оптимальными параметрами. Для Docling и MinerU повторно используются блоки из `PDF_MD_Smoke_Test_1.ipynb`, чтобы иметь единый набор параметров.

### 3. Приведение выгрузок к единому виду

Папка `convert/` содержит конвертеры, каждый из которых нормализует вывод определённого поставщика до целевого Markdown с `<img data-bbox="…">…</img>`:

- `convert_dpsk_ocr.py` — DeepSeek-OCR (парсит `<|ref|>`/`<|det|>` блоки из `_det.mmd`).
- `convert_hy_ocr.py` — Hunyuan OCR (делит Markdown по маркерам страниц и конвертирует pipe-таблицы в HTML).
- `convert_mineru_ppl.py` — MinerU (читает JSON `*_content_list.json`, сортирует блоки и превращает их в Markdown).
- `convert_pp_sv3.py` — PP-Structure V3 (анализирует `<div>`-блоки, вытаскивает изображения/таблицы и переписывает по страницам).
- `convert_z_ai_glm4_6v.py` — GLM-4.6V (убирает `<|begin_of_box|>` сегменты, очищает текст и собирает файл документа).

### 4. Расчёт метрик

`pdf_md_benchmark_doc.py` — основной скрипт подсчёта метрик. Он очищает Markdown, сегментирует страницы, сопоставляет блоки с GT и пишет:

- `eval_root/summary_document.json` и `summary_dataset.json` для каждой модели,
- результаты `bench_docs.csv` и `bench.csv` (копии лежат в корне репозитория),
- отладочные JSON с разбиением страниц.

Описание всех показателей собрано в `metrics.md` (формулы `text_edit_norm_micro`, METEOR, классификация блоков, TEDS, image F1, score-формула). Дополнительно используется `docs_meta.json` — там перечислены теги (`scan`, `table`, `draw`, `rotation`) для документов, чтобы можно было фильтровать статистику.

## Скрипты и примеры запуска

| Скрипт | Назначение | Пример запуска / параметры |
| --- | --- | --- |
| `pdf_to_md_openrouter.py` | Базовый OpenRouter-конвертер для smoke-тестов: рендерит PDF в PNG, вызывает выбранные VLM и собирает Markdown + кропы `<img>` по страницам. | `OPENROUTER_API_KEY=sk-... python pdf_to_md_openrouter.py --pdf_dir ./pdfs --out_dir ./smoke_outputs --models z-ai/glm-4.6v baidu/ernie-4.5-vl-28b-a3b --scale 1.5 --sleep_s 0.5` |
| `df_to_md_openrouter.py` | Хелпер для запуска OpenRouter по табличному списку документов. **Файл отсутствует в текущей ревизии**, поэтому используйте локальную копию или восстановите его рядом со скриптом `pdf_to_md_openrouter.py` перед запуском (интерфейс должен повторять параметры smoke-версии). | После восстановления: `OPENROUTER_API_KEY=sk-... python df_to_md_openrouter.py --help` |
| `pdf_to_md_openrouter_bench.py` | Основной массовый прогон через OpenRouter; параметры аналогичны smoke-версии, но по умолчанию активирован расширенный список моделей и scale=2 для стабильного OCR. | `OPENROUTER_API_KEY=sk-... python pdf_to_md_openrouter_bench.py --pdf_dir ./pdfs --out_dir ./bench_outputs --models qwen/qwen3-vl-30b-a3b-instruct z-ai/glm-4.6v mistralai/ministral-14b-2512` |
| `convert/convert_dpsk_ocr.py` | Нормализация выгрузок DeepSeek-OCR (`*_det.mmd`) → Markdown. Пути настраиваются константами `INPUT_ROOT` / `OUTPUT_ROOT`. | Скопируйте исходные `*_det.mmd` в `input/dpsk-ocr-Gundam/` и выполните `python convert/convert_dpsk_ocr.py`; готовые md появятся в `output/dpsk-ocr-Gundam/`. |
| `convert/convert_hy_ocr.py` | Конвертация Markdown HunyuanOCR (pipe-таблицы → HTML, разбиение по страницам). | `python convert/convert_hy_ocr.py` при наличии исходных `.md` в `input/hy_ocr/`. |
| `convert/convert_mineru_ppl.py` | Преобразование JSON из MinerU (`*_content_list.json`) в Markdown с сохранением страничной структуры. | Сложите все документы в `input/mineru_ppl/<doc>/auto/` и запустите `python convert/convert_mineru_ppl.py`. |
| `convert/convert_pp_sv3.py` | Приведение вывода PP-Structure V3 (изображения и таблицы упаковываются в `<img>`/HTML). | `python convert/convert_pp_sv3.py` (вход: `input/pp_sv3/`; выход: `output/pp_v3/`). |
| `convert/convert_z_ai_glm4_6v.py` | Очистка GLM-4.6V Markdown от служебных `<|begin_of_box|>` сегментов и сборка документа. | `python convert/convert_z_ai_glm4_6v.py` с исходными страницами в `input/z-ai__glm-4_6v/`. |
| `pdf_md_benchmark_doc.py` | Подсчёт doc-level метрик. Ожидает структуру `outputs/ground_truth/<doc>` и `outputs/<model>/<doc>`. | `python pdf_md_benchmark_doc.py --outputs-root ./outputs --eval-root ./eval_out` (результаты будут в `eval_out/bench*.csv`). |

## Размеры используемых моделей

| Модель | Размер |
| --- | --- |
| `tencent/HunyuanOCR` | ≈ 1B параметров |
| `deepseek-ai/DeepSeek-OCR` | ≈ 3B |
| `allenai/olmOCR-2-7B-1025` | ≈ 8B |
| `qwen/qwen3-vl-30b-a3b-instruct` | 30B суммарно, 3B активны (MoE) |
| `baidu/ernie-4.5-vl-28b-a3b` | ≈ 28B суммарно, 3B активны (MoE) |
| `mistralai/ministral-14b-2512` | 14B |
| `z-ai/glm-4.6V` | 10B |

