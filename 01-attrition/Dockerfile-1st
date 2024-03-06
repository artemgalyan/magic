FROM python:3.11-slim AS compile-image
WORKDIR /app

COPY src/webapp /app
COPY data/interim /app/data
COPY models/CatBoostClassifier.pkl /app/models
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

ENV MODEL_PATH='/app/models/CatBoostClassifier.pkl'
ENV DATA_PATH='/app/data/data.csv'
ENV PREFIX_PATH='/app'

EXPOSE 8501
ENTRYPOINT ["python", "-m", "streamlit", "run", "/app/app.py"]
