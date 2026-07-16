FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt && pip install --no-cache-dir -e .

COPY api ./api

# models/ is a volume in docker-compose: build it once via scripts/build_*_models.py
# and mount it in, rather than baking trained artifacts into the image.
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
