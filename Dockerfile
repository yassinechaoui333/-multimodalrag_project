FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt ./
RUN pip install -r requirements-app.txt

RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser rag_runtime.py streamlit_app.py multimodal_rag.ipynb ./
COPY --chown=appuser:appuser .streamlit ./.streamlit
RUN python -c "import json; from pathlib import Path; p=Path('multimodal_rag.ipynb'); n=json.loads(p.read_text()); [c.update(outputs=[], execution_count=None) for c in n['cells'] if c['cell_type']=='code']; p.write_text(json.dumps(n))"
RUN mkdir -p data && chown appuser:appuser data
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
