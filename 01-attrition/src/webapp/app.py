import streamlit as st

from st_pages import add_page_title, show_pages, Page, add_indentation


def path(filename: str) -> str:
    return 'src/webapp/' + filename


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
