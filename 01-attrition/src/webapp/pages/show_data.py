import os

import streamlit as st

from pandas import read_csv, DataFrame


DATA_PATH = os.environ.get('DATA_PATH', 'data/interim/data.csv')

st.set_page_config(page_title='View data')


@st.cache_resource
def load_data() -> DataFrame:
    return read_csv(DATA_PATH, index_col='EmployeeID')


st.header('Data page')

st.subheader('Data view')
data = load_data()
st.dataframe(data)


