FROM mambaorg/micromamba:1.5.6

WORKDIR /app

COPY src/webapp /app
COPY data/interim /app/data
COPY models/CatBoostClassifier.pkl /app/models
COPY environment.yml /app

ENV MODEL_PATH='/app/models/CatBoostClassifier.pkl'
ENV DATA_PATH='/app/data/data.csv'
ENV PREFIX_PATH='/app'

RUN micromamba install -n base -y -f /app/environment.yml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
EXPOSE 8501
ENTRYPOINT ["micromamba", "run", "-n", "base", "streamlit", "run", "/app/app.py"]
