FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY parse_doc.py /app/parse_doc.py
COPY parse_page.py /app/parse_page.py
COPY parse_row.py /app/parse_row.py
COPY parse_table.py /app/parse_table.py
COPY run_attrs_extraction.py /app/run_attrs_extraction.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
