import streamlit as st
import pandas as pd
import numpy as np
import os
from utils import Portfolio, SimulatedAnnealing, GeneticAlgorithm, HillClimbing

import datetime
import altair as alt

class DataView():
    def __init__(self, portfolio_type):
        with st.container():
            st.header(f"Stock simulation")

            col1, col2, col3, col4 = st.columns(4)  # Define 4 columns for all portfolio types

            # Select the correct portfolio object based on the type
            if portfolio_type == "Random":
                portfolio = st.session_state['portfolio']
            elif portfolio_type == "Simulated Annealing":
                portfolio = st.session_state['sa_portfolio']
            elif portfolio_type == "Hill Climbing":
                portfolio = st.session_state['hc_portfolio']
            else:  # Default to Genetic Algorithm
                portfolio = st.session_state['ga_portfolio']

            # ✅ Add this check before calling .evaluate()
            if portfolio is None:
                st.error(f"{portfolio_type} portfolio is not initialized. Run the algorithm first.")
                return


            # Get money over time and daily stock allocations
            money_chart_data = portfolio.evaluate()
            daily_allocations = portfolio.get_daily_allocations()

            # Merge data properly
            merged_data = daily_allocations.copy()
            merged_data["Money"] = money_chart_data
            merged_data["Date"] = portfolio.time_span["Date"].values  # ✅ Ensure Date is added correctly
            merged_data = merged_data.reset_index().rename(columns={"index": "Day"})

            stock_columns = [col for col in daily_allocations.columns if col != "Date"]  # ✅ Remove Date from % list

            # **Fix: Ensure All Stock Percentages & Date are in the Tooltip**
            tooltip_list = [alt.Tooltip("Date:T", title="Date"), "Day:Q", "Money:Q"] + [
                alt.Tooltip(stock + ":Q", title=f"{stock} (%)") for stock in stock_columns
            ]

            # **📈 Money Line Chart with Date & Stock Percentages in Tooltip**
            self.money_chart = (
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
            st.altair_chart(self.money_chart, use_container_width=True)

    def update_chart(self):
        portfolio = st.session_state['portfolio']
        money_chart_data = portfolio.evaluate()
        daily_allocations = portfolio.get_daily_allocations()

        # **Fix: Ensure "Date" is properly included**
        merged_data = daily_allocations.copy()
        merged_data["Money"] = money_chart_data
        merged_data["Date"] = portfolio.time_span["Date"].values  # ✅ Ensure Date is added correctly
        merged_data = merged_data.reset_index().rename(columns={"index": "Day"})

        stock_columns = [col for col in daily_allocations.columns if col != "Date"]  # ✅ Remove Date from % list

        # **Fix: Ensure All Stock Percentages & Date are in the Tooltip**
        tooltip_list = [alt.Tooltip("Date:T", title="Date"), "Day:Q", "Money:Q"] + [
            alt.Tooltip(stock + ":Q", title=f"{stock} (%)") for stock in stock_columns
        ]

        # **📈 Money Line Chart with Date & Stock Percentages in Tooltip**
        updated_money_chart = (
            alt.Chart(merged_data)
            .mark_line(color="black", strokeWidth=2)
            .encode(
                x="Day:Q",
                y="Money:Q",
                tooltip=tooltip_list  # ✅ Fix: Show Date properly
            )
            .interactive()
        )

        # **Update the Chart**
        st.altair_chart(updated_money_chart, use_container_width=True)

st.set_page_config(layout="wide")

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = Portfolio(1000, ["ALL", "A2M"], '2016-01-01', '2017-01-01')

if 'availableStocks' not in st.session_state:
    st.session_state['availableStocks'] = [file.strip('.csv') for file in os.listdir("./archive")]

if 'ga_portfolio' not in st.session_state:
    st.session_state['ga_portfolio'] = None  # Initialize it properly

if 'hc_portfolio' not in st.session_state:
    st.session_state['hc_portfolio'] = None  # Initialize it


st.title("Stock Portfolio Optimization")

c1, c2 = st.columns([5, 5])

with c1:
    with st.container():
        st.subheader("Select option")

        selected_stocks = st.multiselect("Stock used",
            st.session_state['availableStocks'], ["A2M", "ALL"])
        col1, col2 = st.columns([2, 2])

        start_date = col1.date_input("Starting date", datetime.date(2016, 1, 1))
        end_date = col2.date_input("End date", datetime.date(2017, 1, 1))

        portfolio_type = st.radio("Optimization Type", ["Random", "Simulated Annealing", "Genetic Algorithm", "Hill Climbing"])

        if portfolio_type == 'Genetic Algorithm':
            
            num_population = st.slider("Select the number of stating populationo.", 5, 40)

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

                elif portfolio_type == "Genetic Algorithm":
                    st.session_state['ga_portfolio'] = Portfolio(
                        1000, selected_stocks, start_date=start_date, end_date=end_date)
        
                    ga = GeneticAlgorithm(st.session_state['portfolio'])

                    progress_text = "Operation in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)

                    temp_best = ga.run(30, 0.3, 0.9, 0.5, 20)
                    percent_complete = 0
                    with c2:
                        print("Starting Genetic Algorithm")
                        with st.empty():
                            for result in temp_best:
                                percent_complete += (1/20)
                                my_bar.progress(percent_complete, text=progress_text)
                                st.session_state['portfolio'].pf = result
                                # Update the chart with the new result (without re-creating the whole DataView)
                                DataView(portfolio_type)  # Show the final portfolio data view

                elif portfolio_type == "Hill Climbing":
                    st.session_state['hc_portfolio'] = Portfolio(
                        1000, selected_stocks, start_date=start_date, end_date=end_date
                    )

                    hc = HillClimbing(st.session_state['hc_portfolio'])
                    best_pf, _ = hc.optimize()
                    st.session_state['hc_portfolio'].pf = best_pf  # ✅ Update HC portfolio

                    DataView(portfolio_type)  # ✅ Show the graph!



                st.success("Portfolio updated successfully")
