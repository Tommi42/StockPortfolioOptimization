import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import Portfolio, SimulatedAnnealing
import datetime
import altair as alt

class DataView():

    import altair as alt

class DataView():
    def __init__(self, portfolio_type):
        with st.container():
            st.header(f"Stock simulation")

            col1, col2 = st.columns(2)

            # Portfolio total money value over time
            if portfolio_type == "Random":
                value = st.session_state['portfolio'].evaluate().iloc[-1]['Money']
                col1.metric("Random Portfolio", f"{value:.2f}")
            else:
                value = st.session_state['sa_portfolio'].evaluate().iloc[-1]['Money']
                col2.metric("Simulated Annealing", f"{value:.2f}")

            portfolio = st.session_state['portfolio'] if portfolio_type == "Random" else st.session_state['sa_portfolio']
            money_chart_data = portfolio.evaluate()
            daily_allocations = portfolio.get_daily_allocations()

            # **Fix: Ensure "Date" is properly included**
            merged_data = daily_allocations.copy()
            merged_data["Money"] = money_chart_data["Money"]
            merged_data["Date"] = portfolio.time_span["Date"].values  # ✅ Ensure Date is added correctly
            merged_data = merged_data.reset_index().rename(columns={"index": "Day"})

            stock_columns = [col for col in daily_allocations.columns if col != "Date"]  # ✅ Remove Date from % list

            # **Fix: Ensure All Stock Percentages & Date are in the Tooltip**
            tooltip_list = [alt.Tooltip("Date:T", title="Date"), "Day:Q", "Money:Q"] + [
                alt.Tooltip(stock + ":Q", title=f"{stock} (%)") for stock in stock_columns
            ]

            # **📈 Money Line Chart with Date & Stock Percentages in Tooltip**
            money_chart = (
                alt.Chart(merged_data)
                .mark_line(color="black", strokeWidth=2)
                .encode(
                    x="Day:Q",
                    y="Money:Q",
                    tooltip=tooltip_list  # ✅ Fix: Show Date properly
                )
                .interactive()
            )

            # **Show the Chart**
            st.altair_chart(money_chart, use_container_width=True)
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = Portfolio(1000, ["ALL", "A2M"], '2016-01-01', '2017-01-01')

if 'availableStocks' not in st.session_state:
    st.session_state['availableStocks'] = [file.strip('.csv') for file in os.listdir("./archive")]

st.title("Stock Portfolio Optimization")

with st.container():
    st.subheader("Select option")

    selected_stocks = st.multiselect("Stock used",
        st.session_state['availableStocks'], ["A2M", "ALL"])
    col1, col2 = st.columns([2, 2])

    start_date = col1.date_input("Starting date", datetime.date(2016, 1, 1))
    end_date = col2.date_input("End date", datetime.date(2017, 1, 1))

    portfolio_type = st.radio("Optimization Type", ["Random", "Simulated Annealing"])

    if st.button("Run", type="primary"):
        if len(selected_stocks) == 0:
            st.error("Select at least one stock")
        elif start_date >= end_date:
            st.error("Start date must be before end date")
        else:
            st.session_state['portfolio'] = Portfolio(1000, selected_stocks, start_date=start_date, end_date=end_date)

            if portfolio_type == "Simulated Annealing":
                sa = SimulatedAnnealing(st.session_state['portfolio'])
                best_pf, _ = sa.optimize()
                st.session_state['sa_portfolio'] = Portfolio(
                    1000, selected_stocks, start_date=start_date, end_date=end_date
                )
                st.session_state['sa_portfolio'].pf = best_pf  # ✅ Ensure SA portfolio is updated

            st.success("Portfolio updated successfully")
            DataView(portfolio_type)
