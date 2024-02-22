import os
import pickle

from io import StringIO

import shap
import streamlit as st

from pandas import DataFrame, read_csv, Series
from sklearn.base import BaseEstimator
from streamlit_shap import st_shap


MODEL_PATH = os.environ.get('DATA_PATH', 'models/CatBoostClassifier.pkl')
DATA_PATH = os.environ.get('DATA_PATH', 'data/interim/data.csv')

st.set_page_config(layout='wide')


def set_button_pressed_value(value: bool) -> None:
    st.session_state['predicted'] = value


def is_button_pressed() -> bool:
    if 'predicted' not in st.session_state:
        return False
    return st.session_state['predicted']


@st.cache_resource
def load_model() -> BaseEstimator:
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)


@st.cache_data
def load_data() -> DataFrame:
    return read_csv(DATA_PATH, index_col='EmployeeID')


def drop_if_exist(df: DataFrame, *columns: str) -> DataFrame:
    for col in columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


@st.cache_data
def get_shap_values(prepared_data_copy: DataFrame, _model: BaseEstimator) -> shap.Explanation:
    explainer = shap.TreeExplainer(_model[-1])
    shap_values = explainer(_model[:-1].transform(prepared_data_copy))
    shap_values.feature_names = prepared_data_copy.columns
    return shap_values


def make_prediction(df: DataFrame, explain: bool) -> None:
    real_data = load_data()
    model = load_model()
    prepared_data = drop_if_exist(df.copy()[real_data.columns], 'Attrition')
    prepared_data_copy = prepared_data.copy()
    results = model.predict_proba(prepared_data)
    prepared_data.insert(1, 'Attrition probability', Series(data=100*results[:, 1]))
    st.subheader('Results')
    st.dataframe(prepared_data)
    if not explain:
        return
    st.subheader('Explanation with SHAP')
    shap_values = get_shap_values(prepared_data_copy, model)
    st_shap(shap.plots.beeswarm(shap_values, max_display=len(prepared_data_copy.columns)))
    st.subheader('Explain every instance')
    instance_idx = st.number_input('Instance index', min_value=0, max_value=len(prepared_data_copy), step=1)
    explainer = shap.TreeExplainer(model[-1])
    instance = [list(prepared_data_copy.iloc[instance_idx])]
    explanation = explainer(model[:-1].transform(instance))
    explanation.feature_names = prepared_data_copy.columns
    st_shap(shap.plots.waterfall(explanation[0], max_display=len(prepared_data_copy.columns)), width=1600)


def show_dataframe(file: DataFrame) -> None:
    st.dataframe(file)
    explain = st.checkbox('Explain results')
    if st.button('Make prediction') or is_button_pressed():
        set_button_pressed_value(True)
        make_prediction(file, explain)
    else:
        set_button_pressed_value(False)


file = st.file_uploader('Upload your dataframe')
if file is not None:
    content = file.read().decode()
    df = read_csv(StringIO(content))
    show_dataframe(df)
