FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

FROM base AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip wheel --no-deps --wheel-dir /wheels .

FROM base AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /wheels /wheels
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install /wheels/*.whl && pip install . && rm -rf /wheels
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"
CMD ["uvicorn", "agent_service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
