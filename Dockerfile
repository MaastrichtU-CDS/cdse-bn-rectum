FROM python:3.8.1-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app/

RUN python -m pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile


CMD ["python", "main.py"]