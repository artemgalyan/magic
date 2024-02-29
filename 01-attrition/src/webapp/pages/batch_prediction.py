import os
import pickle
from io import StringIO

import numpy as np
import shap
import streamlit as st
from pandas import DataFrame, read_csv, Series
from sklearn.pipeline import Pipeline
from st_aggrid import AgGrid, AgGridReturn, GridOptionsBuilder, DataReturnMode
from streamlit_shap import st_shap

MODEL_PATH = os.environ.get('DATA_PATH', 'models/CatBoostClassifier.pkl')
DATA_PATH = os.environ.get('DATA_PATH', 'data/interim/data.csv')

st.set_page_config(layout='wide')


def show_dataframe(df: DataFrame, selection_mode: str | None = 'single', page_size: int = 20) -> AgGridReturn:
    options_builder = GridOptionsBuilder.from_dataframe(df)
    options_builder.from_dataframe(df)
    options_builder.configure_pagination(True, paginationPageSize=page_size, paginationAutoPageSize=False)
    options_builder.configure_selection(selection_mode=selection_mode, use_checkbox=True)
    return AgGrid(df, gridOptions=options_builder.build(), data_return_mode=DataReturnMode.FILTERED)


def set_button_pressed_value(value: bool) -> None:
    st.session_state['predicted'] = value


def is_button_pressed() -> bool:
    if 'predicted' not in st.session_state:
        return False
    return st.session_state['predicted']


@st.cache_resource
def load_model() -> Pipeline:
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


def get_shap_values(prepared_data_copy: DataFrame, _model: Pipeline) -> shap.Explanation:
    explainer = shap.TreeExplainer(_model[-1])
    shap_values = explainer(_model[:-1].transform(prepared_data_copy))
    shap_values.feature_names = prepared_data_copy.columns
    return shap_values


def explain_predictions(explanation: shap.Explanation, number_of_columns: int) -> None:
    st.subheader('Explanation with SHAP')
    if len(explanation) == 1:
        plot_type = st.radio('Explanation format', options=['Beeswarm', 'Waterfall'], index=len(explanation) > 0)
    else:
        plot_type = 'Beeswarm'
    if plot_type == 'Beeswarm':
        st_shap(shap.plots.beeswarm(explanation, max_display=number_of_columns))
    else:
        st_shap(shap.plots.waterfall(explanation[0], max_display=number_of_columns))


def make_prediction(df: DataFrame, explain: bool) -> None:
    real_data = load_data()
    model = load_model()
    prepared_data = drop_if_exist(df.copy()[real_data.columns], 'Attrition')
    results = model.predict_proba(prepared_data)
    result_column = 'Attrition probability (%)'
    prepared_data.insert(1, result_column, Series(data=np.round(100 * results[:, 1], 1)))
    st.subheader('Results')
    grid_data = show_dataframe(prepared_data, selection_mode='multiple')
    if not explain:
        return
    selected_data = grid_data.data \
        if len(grid_data.selected_rows) == 0 \
        else DataFrame(grid_data.selected_rows).drop(columns=['_selectedRowNodeInfo'])
    selected_data.fillna(value=np.nan, inplace=True)
    explain_predictions(get_shap_values(selected_data.drop(columns=[result_column]), model),
                        len(real_data.columns) - 1)


def show_data(file: DataFrame) -> None:
    selection_data = show_dataframe(file, selection_mode='multiple')
    print(selection_data.selected_rows)
    prediction_data = selection_data.data \
        if len(selection_data.selected_rows) == 0 \
        else DataFrame(selection_data.selected_rows).drop(columns=['_selectedRowNodeInfo'])
    explain = st.checkbox('Explain predictions')
    if st.button('Make prediction') or is_button_pressed():
        set_button_pressed_value(True)
        make_prediction(prediction_data, explain)
    else:
        set_button_pressed_value(False)


file = st.file_uploader('Upload your dataframe')
if file is not None:
    content = file.read().decode()
    df = read_csv(StringIO(content))
    show_data(df)
