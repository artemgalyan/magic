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

page_size = st.slider('Number of rows in a page', min_value=0, max_value=100, value=20)
selection_mode = st.radio('Row selection mode', options=['single', 'multiple'])
data = load_data()
options_builder = GridOptionsBuilder.from_dataframe(data)
options_builder.from_dataframe(data)
options_builder.configure_pagination(True, paginationPageSize=page_size, paginationAutoPageSize=False)
options_builder.configure_selection(selection_mode=selection_mode, use_checkbox=True)
AgGrid(data, gridOptions=options_builder.build())
