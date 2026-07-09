FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY web_app/ web_app/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/demo/ data/demo/
COPY config/ config/
COPY Makefile ./

ENV CLASSIFIER_MODE=heuristic
ENV CLASSIFY_REQUIRE_PANORAMA=0
ENV PYTHONUNBUFFERED=1
ENV FLASK_PORT=8765

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "from web_app.app import app" || exit 1
CMD ["python", "-m", "web_app.app"]
