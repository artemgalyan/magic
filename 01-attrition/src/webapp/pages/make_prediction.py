import os
import pickle

from collections.abc import Generator, Iterable
from typing import TypeVar, ContextManager

import shap
import streamlit as st

from pandas import read_csv, DataFrame
from shap import TreeExplainer
from sklearn.base import BaseEstimator
from streamlit_shap import st_shap

DATA_PATH = os.environ.get('DATA_PATH', 'data/interim/data.csv')
MODEL_PATH = os.environ.get('DATA_PATH', 'models/CatBoostClassifier.pkl')

st.set_page_config(layout='wide')


T = TypeVar('T')


def infinite_loop(iterable: Iterable[T]) -> Generator[T, None, None]:
    while True:
        yield from iterable


@st.cache_data
def load_data() -> DataFrame:
    return read_csv(DATA_PATH, index_col='EmployeeID')


@st.cache_resource
def load_model() -> BaseEstimator:
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)


@st.cache_data
def get_unique_values(df: DataFrame, column: str) -> list[str]:
    return list(df[column].unique())


def read_column(df: DataFrame, column: str, col: ContextManager) -> str | float:
    assert column in df.columns
    with col:
        if df[column].dtype == 'object':
            return st.radio(label=column, options=get_unique_values(df, column))
        granularity = 0.01 if str(df[column].dtype).startswith('float') else 1
        return st.number_input(label=column, step=granularity)


def get_predict_color(value: float) -> str:
    if value < 0.5:
        return 'green'
    if value < 0.8:
        return 'orange'
    return 'red'


def on_predict_clicked(user_input: dict[str, str | float], model: BaseEstimator,
                       explain_model: bool) -> None:
    x = DataFrame(user_input)
    proba = model.predict_proba(x)[0, 1]
    color = get_predict_color(proba)
    st.subheader('Prediction')
    st.markdown(f'Predicted probability of attrition for this person is :{color}[{100*proba:.4f}%].\n')
    if not explain_model:
        return

    explainer = TreeExplainer(model[-1])
    shap_values = explainer(model[:-1].transform(x))
    shap_values.feature_names = x.columns
    st.subheader('Explanation with SHAP')
    st_shap(shap.plots.waterfall(shap_values[0], max_display=len(x.keys())))


data = load_data()
st.header('Make prediction')
st.markdown('Here you can make predictions on given instance. Categorical values can be selected only'
            'from ones that are present in the dataset')

columns = st.columns(2)
user_input = {column_name: [read_column(data, column_name, col)]
              for column_name, col in zip(data.columns, infinite_loop(columns))
              if column_name != 'Attrition'}
explain_model = st.checkbox('Explain prediction', value=True)

if st.button('Make prediction'):
    on_predict_clicked(user_input, load_model(), explain_model)
