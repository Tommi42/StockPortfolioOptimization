import streamlit as st
from dataReader import DataReader

class DataView():

    def __init__(self, dataObj):
        with st.container():
            st.header(f"{dataObj.name}")
            st.metric("Avarage Close", dataObj.avarage_close())


