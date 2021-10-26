FROM python:3.7-slim

RUN apt-get update --fix-missing && apt-get install -y libpq-dev g++

COPY app /app
COPY helpers /helpers
COPY model /model

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5001

CMD ["python", "/app/app.py"]