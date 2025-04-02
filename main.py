import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from utils import Portfolio, SimulatedAnnealing, GeneticAlgorithm, HillClimbing, TabuSearch

import datetime
import altair as alt

class DataView():
    def __init__(self, portfolio_type):
        with st.container():
            st.header(f"Stock simulation")

            col1, col2, col3, col4 = st.columns(4)  

            
            if portfolio_type == "Random":
                portfolio = st.session_state['portfolio']
            elif portfolio_type == "Simulated Annealing":
                portfolio = st.session_state['sa_portfolio']
            elif portfolio_type == "Hill Climbing":
                portfolio = st.session_state['hc_portfolio']
            elif portfolio_type == "Tabu Search":
                portfolio = st.session_state['ts_portfolio']
            else:  
                portfolio = st.session_state['ga_portfolio']

            
            if portfolio is None:
                st.error(f"{portfolio_type} portfolio is not initialized. Run the algorithm first.")
                return

            
            money_chart_data = portfolio.evaluate()
            daily_allocations = portfolio.get_daily_allocations()

            
            final_value = money_chart_data[-1]
            st.metric(label="Final Portfolio Value", value=f"${final_value:,.2f}", delta = f"{(final_value - 1000):,.2f}")

            
            merged_data = daily_allocations.copy()
            merged_data["Money"] = money_chart_data
            merged_data["Date"] = portfolio.time_span["Date"].values  
            merged_data = merged_data.reset_index().rename(columns={"index": "Day"})

            stock_columns = [col for col in daily_allocations.columns if col != "Date"]  

           
            tooltip_list = [alt.Tooltip("Date:T", title="Date"), "Day:Q", "Money:Q"] + [
                alt.Tooltip(stock + ":Q", title=f"{stock} (%)") for stock in stock_columns
            ]

            
            self.money_chart = (
                alt.Chart(merged_data)
                .mark_line(color="red", strokeWidth=2)
                .encode(
                    x="Day:Q",
                    y="Money:Q",
                    tooltip=tooltip_list  
                )
                .interactive()
            )

            
            st.altair_chart(self.money_chart, use_container_width=True)
            st.line_chart(portfolio.pf)

st.set_page_config(layout="wide", page_title='AIPort')


if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = Portfolio(1000, ["ES1"], '2016-02-15', '2016-03-31')

if 'availableStocks' not in st.session_state:
    st.session_state['availableStocks'] = [file.strip('.csv') for file in os.listdir("./archive")]

if 'ga_portfolio' not in st.session_state:
    st.session_state['ga_portfolio'] = st.session_state['portfolio']

if 'hc_portfolio' not in st.session_state:
    st.session_state['hc_portfolio'] = st.session_state['portfolio'] 

if  'sa_portfolio' not in st.session_state:
    st.session_state['sa_portfolio'] = st.session_state['portfolio']

if 'ts_portfolio' not in st.session_state:
    st.session_state['ts_portfolio'] = st.session_state['portfolio']


st.title("Stock Portfolio Optimization")

c1, c2 = st.columns([2, 5])

with c1:
    with st.container():
        st.subheader("Select option")

        selected_stocks = st.multiselect("Stock used",
            st.session_state['availableStocks'], ["ES1", "ES3"])
        col1, col2 = st.columns([2, 2])

        start_date = col1.date_input("Starting date", datetime.date(2016, 2, 1))
        end_date = col2.date_input("End date", datetime.date(2016, 6, 1))

        portfolio_type = st.radio("Optimization Type", ["Random", "Simulated Annealing", "Genetic Algorithm", "Hill Climbing", "Tabu Search"])

        if portfolio_type == 'Genetic Algorithm':
            ga_num_population = st.slider("Starting population.", 5, 40, 15)
            ga_max_iter = st.slider("Max Iteration", 10, 60, 20)
            k_genetic_mutation = st.slider("Genetic Mutation Rate", 0.1, 1.0, 0.3)
            k_modified_elephant = st.slider("Modified Portfolio Rate", 0.1, 1.0, 0.9)
            k_crossover = st.slider("Crossover Rate", 0.1, 1.0, 0.5)

        if portfolio_type == "Simulated Annealing":
            sa_max_iter = st.slider("Max Iterations", 10, 1000, 200)
            sa_temp = st.slider("Initial Temperature", 1.0, 500.0, 100.0)
            sa_cooling = st.slider("Cooling Rate", 0.80, 0.99, 0.95)

        if portfolio_type == "Hill Climbing":
            hc_max_ter = st.slider("Max Iterations", 10, 500, 200)
            hc_step_size = st.slider("Step size", 0.01, 0.7, 0.3)

        if portfolio_type == "Tabu Search":
            ts_max_iter = st.slider("Max Iterations", 10, 1000, 100)
            ts_max_size = st.slider("Max Size", 10, 100, 50)

        if st.button("Run", type="primary"):

            if len(selected_stocks) == 0:
                st.error("Select at least one stock")
            elif start_date >= end_date:
                st.error("Start date must be before end date")
            else:
                st.session_state['portfolio'] = Portfolio(1000, selected_stocks, start_date=start_date, end_date=end_date)

                if portfolio_type == "Simulated Annealing":
                    st.session_state['sa_portfolio'] = st.session_state['portfolio']
                    sa = SimulatedAnnealing(
                        st.session_state['sa_portfolio'],
                        initial_temp=sa_temp,
                        cooling_rate=sa_cooling,
                        max_iter=sa_max_iter
                    )

                    progress_text = "Working on it... (Simulated Annealing ❄️)"
                    with c2:
                        my_bar = st.progress(0, text=progress_text)
                        percent_complete = 0
                        with st.empty():
                            for _ in range(sa_max_iter):
                                sa.step()  # Perform a single SA iteration
                                percent_complete += 1 / sa_max_iter
                                my_bar.progress(percent_complete, text=progress_text)
                                st.session_state['sa_portfolio'].pf = sa.best_pf  # Update best pf
                                DataView(portfolio_type)

                    st.success("Simulated Annealing optimization completed!")
   
                elif portfolio_type == "Genetic Algorithm":
                    st.session_state['ga_portfolio'] = st.session_state['portfolio']
        
                    ga = GeneticAlgorithm(
                        st.session_state['portfolio'],
                        k_genetic_mutation=k_genetic_mutation,
                        k_modified_elephant=k_modified_elephant,
                        k_crossover=k_crossover
                    )

                    progress_text = "Working on it... (I'm not Warren Buffett but I'm doing my best)"
                    with c2: my_bar = st.progress(0, text=progress_text)
                    temp_best = ga.optimize(ga_num_population, ga_max_iter)
                    percent_complete = 0
                    with c2:
                        with st.empty():
                            for result in temp_best:
                                percent_complete = percent_complete + (1/ga_max_iter)
                                my_bar.progress(percent_complete, text=progress_text)
                                st.session_state['ga_portfolio'].pf = result
                                DataView(portfolio_type)

                elif portfolio_type == "Hill Climbing":

                    progress_text = "Working on it... (I'm not Warren Buffett but I'm doing my best)"
                    with c2: my_bar = st.progress(0, text=progress_text)
                    st.session_state['hc_portfolio'] = st.session_state['portfolio']
                    hc = HillClimbing(st.session_state['hc_portfolio'], iterations=hc_max_ter, step_size=hc_step_size)

                    best_pf = hc.optimize()
                    percent_complete = 0
                    with c2:
                        with st.empty():
                            for result in best_pf:
                                percent_complete = percent_complete + (1/hc_max_ter)
                                my_bar.progress(percent_complete, text=progress_text)
                                st.session_state['hc_portfolio'].pf = result
                                DataView(portfolio_type)
                        
                           
                            st.session_state['hc_portfolio'].pf = best_pf  # ✅ Update HC portfolio
                            DataView(portfolio_type)  # ✅ Show the graph!
                
                elif portfolio_type == "Tabu Search":
                    st.session_state['ts_portfolio'] = st.session_state['portfolio']
                    with c2:
                        progress_text = "Working on it..."
                        with c2: my_bar = st.progress(0, text=progress_text)
                        percent_complete = 0

                        ts = TabuSearch(st.session_state['ts_portfolio'], iterations=ts_max_iter, tabu_size=ts_max_size)
                        with st.empty():
                            print("Optimising")
                            for result in ts.optimize():
                                percent_complete = percent_complete + (1/ts_max_iter)
                                my_bar.progress(percent_complete, text=progress_text)
                                st.session_state['ts_portfolio'].pf = result
                                DataView(portfolio_type)
                        
                else:
                    st.session_state['portfolio'] = Portfolio(1000, selected_stocks, start_date=start_date, end_date=end_date)
                    with c2: DataView('Random')

                st.success("Portfolio updated successfully")
                if random.random() > 0.8:
                    st.balloons()
