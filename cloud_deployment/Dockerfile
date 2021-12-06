FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_eta=0.1_max_depth=5_min_ch_w=5.bin", "./"]

CMD gunicorn --bind 0.0.0.0:$PORT predict:app