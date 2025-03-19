import streamlit as st
from dataReader import DataReader
import pandas as pd
import numpy as np

class DataView():

    def __init__(self, dataObj):
        with st.container():
            st.header(f"30 Stock simulation")
            col1, col2, col3 = st.columns(3)

            col1.metric("HILL - CLIMBING", "+113.5", "+113.5")
            col2.metric("SIMULATED ANNEALING", "+50.8", "-8%")
            col3.metric("GENETIC ALGORITHM", "+167.1", "4%")


            chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

            st.line_chart(chart_data)


