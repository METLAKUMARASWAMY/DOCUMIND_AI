# ---------------------------------------------------------------------------
# Stage 1: Tesseract, pinned to the version the project targets.
#
# Debian trixie packages Tesseract 5.5.0. The project targets 5.5.3, and the
# difference is not cosmetic: on a 1976 scan in the regression corpus, 5.5.0
# reads a required field as "No" where 5.5.3 reads it correctly, which is the
# whole of the remaining Playwright failure. Rather than weaken the assertion
# to match the weaker engine, the engine is built from source at the pinned
# version, and the language data is pinned by checksum to the exact files the
# development host uses (tessdata_fast, not the default tessdata — they are
# different models and produce different output).
#
# Measured build cost: ~75 seconds. Everything below is content-addressed, so
# an upstream change fails the build loudly instead of silently shifting OCR.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS tesseract-build

ARG TESSERACT_VERSION=5.5.3
ARG TESSDATA_REF=4.1.0
ARG TESSDATA_ENG_SHA256=7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2
ARG TESSDATA_OSD_SHA256=9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential autoconf automake libtool pkg-config ca-certificates curl \
       libleptonica-dev libpng-dev libjpeg-dev libtiff-dev zlib1g-dev libwebp-dev \
       libarchive-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/tesseract.tar.gz \
      "https://github.com/tesseract-ocr/tesseract/archive/refs/tags/${TESSERACT_VERSION}.tar.gz" \
 && tar -xzf /tmp/tesseract.tar.gz -C /tmp \
 && cd "/tmp/tesseract-${TESSERACT_VERSION}" \
 && ./autogen.sh \
 && ./configure --prefix=/opt/tesseract --disable-openmp --disable-graphics --disable-doc \
 && make -j"$(nproc)" \
 && make install \
 && /opt/tesseract/bin/tesseract --version

# Language data, verified against the checksums the host resolves to. A
# mismatch here means the container would read documents differently from the
# machine the regression corpus was labelled on, so it must stop the build.
RUN set -eux; \
    for pair in "eng:${TESSDATA_ENG_SHA256}" "osd:${TESSDATA_OSD_SHA256}"; do \
      name="${pair%%:*}"; sum="${pair##*:}"; \
      curl -fsSL -o "/opt/tesseract/share/tessdata/${name}.traineddata" \
        "https://github.com/tesseract-ocr/tessdata_fast/raw/${TESSDATA_REF}/${name}.traineddata"; \
      echo "${sum}  /opt/tesseract/share/tessdata/${name}.traineddata" | sha256sum -c -; \
    done

# ---------------------------------------------------------------------------
# Stage 2: the application image.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/tesseract/bin:$PATH \
    LD_LIBRARY_PATH=/opt/tesseract/lib \
    TESSDATA_PREFIX=/opt/tesseract/share/tessdata

WORKDIR /app

# Runtime libraries only — the toolchain stays in the builder stage. These are
# what the compiled Tesseract links against; `tesseract --version` in the
# healthcheck below would fail if any were missing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl libgl1 libglib2.0-0 \
       libleptonica6 libpng16-16 libjpeg62-turbo libtiff6 libwebp7 libwebpmux3 \
       libgif7 libopenjp2-7 libarchive13 libcurl4 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tesseract-build /opt/tesseract /opt/tesseract

# Fail fast at build time if the pinned engine is not the one that landed.
RUN tesseract --version | head -1 | grep -q "5.5.3" \
    && tesseract --list-langs

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
