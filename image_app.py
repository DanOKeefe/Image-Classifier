import streamlit as st
import image_upload_page
import image_url_page

PAGES = {
    "Image URL": image_url_page,
    "Image Upload": image_upload_page
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()