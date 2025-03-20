import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import Portfolio
import datetime

class DataView():

    def __init__(self):
        with st.container():
            st.header(f"Stock simulation")
            col1, col2, col3 = st.columns(3)
            col1.metric("HILL - CLIMBING", "+113.5", "+113.5")
            col2.metric("SIMULATED ANNEALING", "+50.8", "-8%")
            col3.metric("GENETIC ALGORITHM", "+167.1", "4%")


            stock_chart_data = st.session_state['portfolio'].get_portfolio_sampled(round(42))
            st.line_chart(stock_chart_data, x_label =None)

            money_chet_daya = st.session_state['portfolio'].evaluate()
            st.line_chart(money_chet_daya, x_label =None)


            

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = Portfolio(1000, ["ALL", "A2M"], '2016-01-01', '2017-01-01')

if 'availableStocks' not in st.session_state:
    st.session_state['availableStocks'] =[file.strip('.csv') for file in os.listdir("./archive")]


st.title("Stock Portfolio Optimization")

with st.container():
    st.subheader("Select option")
    
    selected_stocks = st.multiselect("Stock used",
        st.session_state['availableStocks'],  ["A2M", "ALL"])
    col1, col2= st.columns([2, 2])

    start_date =  col1.date_input("Starting date", datetime.date(2016, 1, 1))
    end_date = col2.date_input("End date", datetime.date(2017, 1, 1))

   
    
    if st.button("Run", type="primary"):
        if len(selected_stocks) == 0:
            st.error("Select at least one stock")
        else:
            st.session_state['portfolio'] = Portfolio(1000, selected_stocks, start_date=start_date, end_date=end_date)

a = DataView()

