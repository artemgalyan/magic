import os

import streamlit as st

from pandas import read_csv, DataFrame
from st_aggrid import AgGrid, GridOptionsBuilder


DATA_PATH = os.environ.get('DATA_PATH', 'data/interim/data.csv')

st.set_page_config(layout='wide')


@st.cache_data
def load_data() -> DataFrame:
    return read_csv(DATA_PATH, index_col='EmployeeID')


st.header('Data page')

row_number = st.slider('Number of rows', min_value=0, max_value=100, value=20)
data = load_data()
AgGrid(data.head(row_number),
       gridOptions=GridOptionsBuilder.from_dataframe(data).build())
