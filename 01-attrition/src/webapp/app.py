import os

from pathlib import Path

import streamlit as st

from st_pages import add_page_title, show_pages, Page, add_indentation


PREFIX_PATH = Path(os.environ.get('PREFIX_PATH', 'src/webapp/'))


def path(filename: str) -> str:
    return str(PREFIX_PATH / filename)


add_page_title()

show_pages([
    Page(path('app.py'), 'Home', '🏠'),
    Page(path('pages/show_data.py'), 'Show data', '📄'),
    Page(path('pages/make_prediction.py'), 'Make prediction', '🧍‍♂️'),
    Page(path('pages/batch_prediction.py'), 'Make batch prediction', '👯‍️'),
])

add_indentation()

st.markdown('Hi👋 Here you can explore our model and data and make some predictions. '
            'You can find everything on the sidebar on the left side!🤓')
