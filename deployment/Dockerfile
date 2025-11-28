# Python base image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user     PATH=/home/user/.local/bin:$PATH

WORKDIR /app

COPY --chown=user:user deployment/app.py ./app.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
