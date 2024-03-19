FROM python:3.11.7-slim AS compile-image
WORKDIR /app

COPY requirements.txt /app
RUN apt-get update
RUN pip install --target=/app/dependencies -r /app/requirements.txt

FROM python:3.11.7-slim as release
WORKDIR /app
COPY --from=compile-image /app /app

COPY src/webapp /app
COPY data/interim /app/data
COPY models/CatBoostClassifier.pkl /app/models/CatBoostClassifier.pkl

ENV MODEL_PATH='/app/models/CatBoostClassifier.pkl'
ENV DATA_PATH='/app/data/data.csv'
ENV PREFIX_PATH='/app'
ENV PYTHONPATH="$PYTHONPATH:/app/dependencies"

EXPOSE 8501
ENTRYPOINT ["python", "-m", "streamlit", "run", "/app/app.py", "--global.developmentMode", "false", "--server.port", "8501"]