FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential libffi-dev libssl-dev python3-dev && \
    pip install --no-cache-dir -r /app/requirements.txt && apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY . /app
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port","8501","--server.address","0.0.0.0"]