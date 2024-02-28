import os

from enum import Enum

import streamlit as st
import plotly.express as px

from pandas import read_csv, DataFrame
from plotly.graph_objects import Figure
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode


class FeatureType(Enum):
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'


def histogram_wrapper(df: DataFrame, x: str, y: str) -> Figure:
    return px.histogram(df, x=x, color=y, histnorm='probability density')


DATA_PATH = os.environ.get('DATA_PATH', 'data/interim/data.csv')

st.set_page_config(layout='wide')


@st.cache_data
def load_data() -> DataFrame:
    return read_csv(DATA_PATH, index_col='EmployeeID')


st.header('Data page')

page_size = st.slider('Number of rows in a page', min_value=0, max_value=100, value=20)
data = load_data()
options_builder = GridOptionsBuilder.from_dataframe(data)
options_builder.from_dataframe(data)
options_builder.configure_pagination(True, paginationPageSize=page_size, paginationAutoPageSize=False)
returned = AgGrid(data, gridOptions=options_builder.build(), data_return_mode=DataReturnMode.FILTERED)

columns = list(data.columns)
numerical_features = {col for col in data.columns if data[col].dtype != object}
categorical_features = set(data.columns) - numerical_features

x_axis = st.selectbox('X axis', columns)
y_axis = st.selectbox('Y axis', columns)

display_types = {
    (FeatureType.NUMERICAL, FeatureType.NUMERICAL): px.scatter,
    (FeatureType.NUMERICAL, FeatureType.CATEGORICAL): px.box,
    (FeatureType.CATEGORICAL, FeatureType.NUMERICAL): px.box,
    (FeatureType.CATEGORICAL, FeatureType.CATEGORICAL): histogram_wrapper
}

x_feature_type = FeatureType.NUMERICAL if x_axis in numerical_features else FeatureType.CATEGORICAL
y_feature_type = FeatureType.NUMERICAL if y_axis in numerical_features else FeatureType.CATEGORICAL

st.plotly_chart(display_types[x_feature_type, y_feature_type](data, x=x_axis, y=y_axis), use_container_width=True)
