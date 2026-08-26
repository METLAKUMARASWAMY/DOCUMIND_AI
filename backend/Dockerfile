FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is a thin indirection (-r requirements/dev.txt), so the
# requirements/ directory has to come with it or the install cannot resolve.
COPY requirements.txt pyproject.toml ./
COPY requirements ./requirements

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
