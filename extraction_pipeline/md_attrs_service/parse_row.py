import os
import requests
from time import sleep
import json
from pathlib import Path
from time import time
from bs4 import BeautifulSoup

from tqdm import tqdm
from json import JSONDecodeError


def openrouter_call(messages, schema, api_key, openrouter_model, temperature,
                    max_tokens, cooldown=0):
    s_time = time()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": openrouter_model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
            "plugins": [
                {"id": "response-healing"}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "provider": {"require_parameters": True}
        },
    )
    response_wait = time() - s_time
    # print('Spent', response_wait, 's')
    if cooldown:
        sleep(cooldown)

    return response


def extract_requirements_from_prompt(
    prompt_content,
    schema,
    api_key,
    openrouter_model,
    temperature=0.0,
    max_tokens=1000,
    cooldown=0.5,
):
    messages = [{"role": "user", "content": prompt_content}]
    response = openrouter_call(
        messages=messages,
        schema=schema,
        api_key=api_key,
        openrouter_model=openrouter_model,
        temperature=temperature,
        max_tokens=max_tokens,
        cooldown=cooldown,
    )
    data = response.json()
    try:
        return json.loads(data["choices"][0]["message"]["content"])["requirements"]
    except KeyError:
        print(data.get("error", data))
        sleep(60)
    except JSONDecodeError:
        print("JSONDecodeError")
        sleep(60)
    return []


ROW_IE_PROMPT_TEMPLATE = """Извлеки характеристики и требования из опросного листа (технического документа на оборудование).

На входе: часть таблицы HTML, полученная из PDF.
На выходе: спсиок требований в JSON формате. Без комментариев, без пояснений, без Markdown.

JSON format:
{{
  "requirements": [
    {{
      "name": "<название параметра, которое запрашивает Заказчик>",
      "request_value": "<значение парметра, которое запрашивает Заказчик>",
    }},
    ...
  ]
}}

СТРОГИЕ ПРАВИЛА:

1) Верни ТОЛЬКО JSON. Любой текст вне JSON запрещён.

2) name:
   - Используй формулировку характеристики или требования из документа.
   - Нормализуй переносы строк и пробелы, но НЕ меняй смысл.
   - Если требование оформлено предложением (например: «… должен …»), используй ВЕСЬ текст требования как name, в одну строку.

3) request_value:
   - Для параметров, заданных в таблицах или структурированных блоках, указывай значение, сопоставленное этому параметру в документе.
   - Для декларативных требований, не имеющих явного значения, используй подтверждающее значение, например: «Да» или «Подтверждаю» (если явно указано в документе).
   - Значения вида:
       «Определяет завод-изготовитель»,
       «Уточняется согласно рабочей документации»,
       «По таблице …»,
       «___», «____»
     считаются валидными значениями и должны сохраняться без изменений.
   - Если параметр содержит несколько подзначений, объединяй их в ОДНУ строку через «; », сохраняя все числа, единицы измерения и ссылки.

4) НЕ добавляй атрибуты, которых нет в предоставленной части таблицы.
5) НЕ изменяй единицы измерения, знаки «−», «+», десятичные разделители.
6) Игнорируй подписи, реквизиты, контактную информацию, номер страницы и т.д. — извлекай только технические характеристики и требования.
7) Не оставляй несколько требований в одном. Если они оформлены в одной ячейке таблицы или одним абзацем, разбей их в отдельные.
8) Верни все характеристики из таблиц страниц, даже исли они представленны частично.

Содержание таблицы:
```
{html_block}
```
"""

TEXT_IE_PROMPT_TEMPLATE = """Извлеки характеристики и требования из опросного листа (технического документа на оборудование).

На входе: отрывок текста, полученный из PDF.
На выходе: спсиок требований в JSON формате. Без комментариев, без пояснений, без Markdown.

JSON format:
{{
  "requirements": [
    {{
      "name": "<название параметра, которое запрашивает Заказчик>",
      "request_value": "<значение парметра, которое запрашивает Заказчик>",
    }},
    ...
  ]
}}

СТРОГИЕ ПРАВИЛА:

1) Верни ТОЛЬКО JSON. Любой текст вне JSON запрещён.

2) name:
   - Используй формулировку характеристики или требования из документа.
   - Нормализуй переносы строк и пробелы, но НЕ меняй смысл.
   - Если требование оформлено предложением (например: "… должен …"), используй ВЕСЬ текст требования как name, в одну строку.

3) request_value:
   - Для параметров, заданных в таблицах или структурированных блоках, указывай значение, сопоставленное этому параметру в документе.
   - Для декларативных требований, не имеющих явного значения, используй подтверждающее значение "Да".
   - Значения вида:
       "Определяет завод-изготовитель",
       "Уточняется согласно рабочей документации",
       "По таблице …",
       "___", "____"
     считаются валидными значениями и должны сохраняться без изменений.
   - Если параметр содержит несколько подзначений, объединяй их в ОДНУ строку через "; ", сохраняя все числа, единицы измерения и ссылки.
   - Для формы с чек-боксами укажи только выбранные варианты

4) НЕ добавляй атрибуты, которых нет в документе.
5) НЕ изменяй единицы измерения, знаки «−», «+», десятичные разделители.
6) Игнорируй титульный лист, подписи, реквизиты, контактную информацию — извлекай только технические характеристики и требования.
7) Не оставляй несколько требований в одном. Если они оформлены в одной ячейке таблицы или одним абзацем, разбей их в отдельные.
8) Верни все характеристики из текста страниц, даже исли они представленны частично.
9) Игнорируй справочные таблицы в приложениях, со сводными значениями/прогнозами без требований или характеристик относящихся к какой-либо продукции.

ТЕКСТ ДОКУМЕНТА:
<<<DOCUMENT

{window_text}

DOCUMENT>>>
"""

SPLIT_CHECK_PROMPT_TEMPLATE = """Decide whether two characteristics are actually one requirement split across neighboring pages.

Return STRICT JSON object only.

Previous window last characteristic:
{prev_item}

Next window first characteristic:
{next_item}

Rules:
1) Return {{\"result\": \"one splitted\"}} if second item clearly continues/completes the first.
2) Return {{\"result\": \"two different\"}} if they are independent rows/requirements.
"""

JOIN_SPLIT_PROMPT_TEMPLATE = """Merge two characteristics that are confirmed to be one split requirement.

Return STRICT JSON object only with fields:
- name
- class

Item A:
{prev_item}

Item B:
{next_item}

Rules:
1) Build one coherent merged requirement while preserving meaning and key numbers/units/references.
"""

TABLE_RELEVANCE_PROMPT_TEMPLATE = """Ты классификатор таблиц из технического документа.

Задача: определить, содержит ли таблица ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ/ХАРАКТЕРИСТИКИ (то, что нужно извлекать как требования),
или это СПРАВОЧНАЯ/СВОДНАЯ таблица (приложение, легенда, расшифровки, сводная таблица значений, обозначения, шапки/реквизиты и т.п.).

Верни СТРОГО JSON без текста вне JSON:
{{
  "result": "requirements" | "reference",
  "reason": "<короткая причина в 3-12 слов>"
}}

Критерии:
- "requirements": есть параметры/характеристики/значения/нормы/должен/не менее/не более/ед.изм/диапазоны/табличные ТТХ предъявляемые для продукции.
- "reference": нет требований; таблица про реквизиты, состав приложений, перечни, условные обозначения, легенды, нумерация, ссылки на документы и т.д.

HTML таблицы:
```html
{table_html}
````

"""

APPENDIX_START_PROMPT_TEMPLATE = """Ты классификатор страниц технического документа.

Задача: определить, начинается ли на этой странице любой раздел приложений.

Считай, что раздел приложений начался, если на странице есть заголовок/маркер вида:
- "Приложение"
- "Приложение 1" (или другой номер)
- "Приложение А" (или другая буква)
- "Appendix"

Верни СТРОГО JSON без текста вне JSON:
{{
  "result": "appendix_start" | "not_appendix_start",
  "reason": "<короткая причина в 3-12 слов>"
}}

Правила:
- Если есть явный заголовок приложения, верни "appendix_start".
- Если это просто упоминание слова "приложение" внутри текста без заголовка начала раздела, верни "not_appendix_start".

Текст страницы:
<<<PAGE
{page_text}
PAGE>>>
"""


IE_RESPONSE_SCHEMA = {
    "name": "requirements_list",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "description": "List of extracted requirements from the document.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Required parameter name or full requirement wording"
                        },
                        "request_value": {
                            "type": "string",
                            "description": (
                                "Required parameter value or \"Да\" "
                                "if the value is not specified "
                                "and the wording of the name simply requires conformity"
                            )
                        },
                    },
                    "required": ["name", "request_value"],
                    "additionalProperties": False
                },
                
            },
        },
        "required": ["requirements"],
        "additionalProperties": False
    },
}

SPLIT_DECISION_SCHEMA = {
    "name": "split_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "result": {"type": "string", "enum": ["one splitted", "two different"]},
        },
        "required": ["result"],
        "additionalProperties": False,
    },
}

MERGED_ITEM_SCHEMA = {
    "name": "merged_item",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Required parameter name or full requirement wording"
            },
            "request_value": {
                "type": "string",
                "description": (
                    "Required parameter value or \"Да\" "
                    "if the value is not specified "
                    "and the wording of the name simply requires conformity"
                )
            },
        },
        "required": ["name", "request_value"],
        "additionalProperties": False,
    },
}

TABLE_RELEVANCE_SCHEMA = {
    "name": "table_relevance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "enum": ["requirements", "reference"],
                "description": "Does table contain requirements (need to parse) or reference (need to skip)"
            },
            "reason": {"type": "string"},
    },
        "required": ["result", "reason"],
        "additionalProperties": False,
    },
}

APPENDIX_START_SCHEMA = {
    "name": "appendix_start",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "enum": ["appendix_start", "not_appendix_start"],
                "description": "Is this page the beginning of appendices section"
            },
            "reason": {"type": "string"},
        },
        "required": ["result", "reason"],
        "additionalProperties": False,
    },
}



def split_table_into_blocks(table):
    rows = table.find_all('tr')
    blocks = []
    active_rowspans = []
    cur_rows = []

    for row in rows:
        # start new block if nothing is currently spanning from previous rows
        if not active_rowspans and cur_rows:
            blocks.append("<table>\n" + "\n".join(cur_rows) + "\n</table>")
            cur_rows = []

        cur_rows.append(str(row))

        # decrement existing spans (we just consumed one row)
        active_rowspans = [r - 1 for r in active_rowspans if (r - 1) > 0]

        # add new spans starting in this row
        for cell in row.find_all(['td', 'th'], recursive=False):
            rs = cell.get('rowspan')
            if rs:
                rs = int(rs)
                if rs > 1:
                    active_rowspans.append(rs - 1)

        # if no spans remain, close block
        if not active_rowspans and cur_rows:
            blocks.append("<table>\n" + "\n".join(cur_rows) + "\n</table>")
            cur_rows = []

    if cur_rows:
        blocks.append("<table>\n" + "\n".join(cur_rows) + "\n</table>")

    return blocks


api_key = os.environ["OPENROUTER_API_KEY"]
openrouter_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3.2")

input_root = Path(os.getenv("OCR_MD_DIR", "./ocr_md/"))
for doc in input_root.glob('*'):
    doc_md_path = doc / (doc.stem + '.md')
    if not doc_md_path.exists():
        continue

    print('Processing for', doc_md_path.name)
    tables_items = {}
    usage = 0
    pages = sorted([p.name for p in doc.glob('page*.md')])
    for page_i, page_name in tqdm(enumerate(pages)):
        page_md_path = doc / page_name
        with open(page_md_path) as f:
            page_content = f.read()

        # Stop extraction when appendices section starts.
        appendix_messages = [
            {
                "role": "user",
                "content": APPENDIX_START_PROMPT_TEMPLATE.format(
                    page_text=page_content[:12000]
                )
            }
        ]
        appendix_response = openrouter_call(
            messages=appendix_messages,
            schema=APPENDIX_START_SCHEMA,
            api_key=api_key,
            openrouter_model=openrouter_model,
            temperature=0.0,
            max_tokens=120,
            cooldown=0.2
        )
        try:
            appendix_verdict = json.loads(
                appendix_response.json()["choices"][0]["message"]["content"]
            )
            if appendix_verdict["result"] == "appendix_start":
                print("APPENDIX START DETECTED:", appendix_verdict["reason"])
                print("Stop extraction for this document at page", page_name)
                break
        except Exception as e:
            print("APPENDIX CHECK ERROR:", e)
            
        soup = BeautifulSoup(page_content, "html.parser")
        tables = soup.findAll('table')
        page_items = []
        for table in tables:
            table_html = str(table)
            messages = [
                {
                    "role": "user",
                    "content": TABLE_RELEVANCE_PROMPT_TEMPLATE.format(
                        table_html=table_html
                    )
                }
            ]
            response = openrouter_call(
                messages=messages,
                schema=TABLE_RELEVANCE_SCHEMA,
                api_key=api_key,
                openrouter_model=openrouter_model,
                temperature=0.0,
                max_tokens=120,
                cooldown=0.2
            )

            try:
                verdict = json.loads(response.json()["choices"][0]["message"]["content"])
                if verdict["result"] == "reference":
                    print(table_html)
                    print("SKIP TABLE:", verdict["reason"])
                    continue
            except Exception as e:
                print("TABLE CLASSIFY ERROR:", e)
                pass
            
            blocks = split_table_into_blocks(table)
            for html_block in blocks:
                page_items += extract_requirements_from_prompt(
                    prompt_content=ROW_IE_PROMPT_TEMPLATE.format(
                        html_block=html_block
                    ),
                    schema=IE_RESPONSE_SCHEMA,
                    api_key=api_key,
                    openrouter_model=openrouter_model,
                    temperature=0.0,
                    max_tokens=1000,
                    cooldown=0.5,
                )

        # Extract from non-table text blocks (same response schema).
        text_soup = BeautifulSoup(page_content, "html.parser")
        for table in text_soup.find_all("table"):
            table.decompose()
        page_text = "\n".join(
            line.strip() for line in text_soup.get_text("\n").splitlines() if line.strip()
        )
        if page_text:
            page_items += extract_requirements_from_prompt(
                prompt_content=TEXT_IE_PROMPT_TEMPLATE.format(
                    window_text=page_text
                ),
                schema=IE_RESPONSE_SCHEMA,
                api_key=api_key,
                openrouter_model=openrouter_model,
                temperature=0.0,
                max_tokens=10000,
                cooldown=0.5,
            )

        if page_items:
            tables_items[page_i] = page_items

    # Decide/merge split requirements between neighboring tables
    items = []
    table_ids = sorted(list(tables_items.keys()))
    for i in range(len(table_ids) - 1):
        cur_table_id = table_ids[i]
        next_table_id = table_ids[i + 1]

        try:
            prev_item = tables_items[cur_table_id][-1]
        except IndexError:
            continue

        try:
            next_item = tables_items[next_table_id][0]
        except IndexError:
            continue

        print('CHECK FOR SPLIT')
        print(prev_item)
        print('---')
        print(next_item)

        # Decide if boundary items are one split requirement or two different
        messages = [
            {
                "role": "user",
                "content": SPLIT_CHECK_PROMPT_TEMPLATE.format(
                    prev_item=json.dumps(prev_item, ensure_ascii=False),
                    next_item=json.dumps(next_item, ensure_ascii=False),
                )
            }
        ]
        response = openrouter_call(
            messages=messages,
            schema=SPLIT_DECISION_SCHEMA,
            api_key=api_key,
            openrouter_model=openrouter_model,
            temperature=0.0,
            max_tokens=100
        )
        decision = json.loads(response.json()["choices"][0]["message"]["content"])["result"]

        if decision == "one splitted":
            # Merge split items into one
            messages = [
                {
                    "role": "user",
                    "content": JOIN_SPLIT_PROMPT_TEMPLATE.format(
                        prev_item=json.dumps(prev_item, ensure_ascii=False),
                        next_item=json.dumps(next_item, ensure_ascii=False),
                    )
                }
            ]
            response = openrouter_call(
                messages=messages,
                schema=MERGED_ITEM_SCHEMA,
                api_key=api_key,
                openrouter_model=openrouter_model,
                temperature=0.0,
                max_tokens=1500
            )
            merged_item = json.loads(response.json()["choices"][0]["message"]["content"])
            print('MERGED')
            print(merged_item)

            # Replace last item in current table and drop first item of next table
            tables_items[cur_table_id][-1] = merged_item
            tables_items[next_table_id] = tables_items[next_table_id][1:]

    # Join all tables to one list (in order)
    for table_id in table_ids:
        for it in tables_items.get(table_id, []):
            items.append(it)

    print('Tokens generated', usage)
    print(len(items), 'were extracted')

    with open(doc_md_path.with_suffix('').as_posix() + '_row.json', 'w') as f:
        f.write(json.dumps(items, ensure_ascii=False))
