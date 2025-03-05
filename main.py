import streamlit as st
from dataReader import DataReader
from stDataView import DataView



st.title("Stock Portfolio Optimization")
d = DataReader('archive/ALL.csv')
a = DataView(d)

