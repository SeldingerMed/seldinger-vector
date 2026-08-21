FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.source="https://github.com/SeldingerMed/seldinger-vector" \
    org.opencontainers.image.description="Vector Cloud control plane and RunPod worker"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[cloud]" \
    && useradd --create-home --uid 10001 vector

USER vector
ENTRYPOINT ["surgeval"]
CMD ["cloud", "worker"]
