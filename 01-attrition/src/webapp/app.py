import streamlit as st

from st_pages import add_page_title, show_pages, Page, add_indentation


def path(filename: str) -> str:
    return 'src/webapp/' + filename


add_page_title()

show_pages([
    Page(path('app.py'), 'Home', '🏠'),
    Page(path('pages/show_data.py'), 'Show data', '📄'),
    Page(path('pages/make_prediction.py'), 'Make prediction', '🧍‍♂️'),
])

add_indentation()

st.markdown('It is the main page. ')
