import csv
import datetime
import pandas as pd
from random import randint
from typing import List 
import matplotlib.pyplot as plt
import numpy as np
import random
import os

class DataReader():
    def __init__(self, data_csv, start_date, end_date):
        self.name = data_csv.strip('.csv').split('/')[1]
        self.data = pd.read_csv(data_csv)
        self.start_date = start_date
        self.end_date = end_date
        self.filtered_data = None
        self.bool_filtered_data = False
                
    def __str__(self):
        return f"--- Data of {self.name} stock ---\n{str(self.data)}"
        
    def get_adj_close(self):
        if not self.bool_filtered_data:
            self.bool_filtered_data = True
            start_date = pd.to_datetime(self.start_date) if self.start_date else pd.to_datetime('1900-01-01')
            end_date = pd.to_datetime(self.end_date) if self.end_date else pd.to_datetime('2100-01-01')
            try:
                self.filtered_data = self.data[(pd.to_datetime(self.data['Date']) > start_date) & 
                                               (pd.to_datetime(self.data['Date']) < end_date)][['Date', 'Adj Close']]
            except Exception as e:
                print("Error with timespan. Probably not all stock selected have data for the selected timespan")
                print(f"{e}")
        return self.filtered_data

class Portfolio:

    def __init__(self, start_value: int, stocks: List[str], start_date, end_date):
        self.start_value = start_value
        self.stockData = {stock: DataReader(f"./archive/{stock}.csv", start_date, end_date) for stock in stocks}
        self.num_day = len(self.stockData[stocks[0]].get_adj_close()['Date'])
        dates_list = [pd.to_datetime(self.stockData[stock].get_adj_close()['Date']) for stock in stocks]
        common_dates = set(dates_list[0]).intersection(*dates_list[1:])
        common_dates = sorted(list(common_dates)) 

        self.time_span = pd.DataFrame({"Date": common_dates})
        self.num_day = len(self.time_span)

        # Initialize stock weights randomly
        pf = pd.DataFrame({stock: [randint(1, 100) for _ in range(len(self.time_span))] for stock in stocks})

        # Normalize weights so that each row sums to 1
        self.pf = pf.div(pf.sum(axis=1), axis=0)

    def get_daily_allocations(self):
        """Returns the percentage of each stock per day in a tooltip-friendly format."""
        daily_allocations = self.pf.copy()
        daily_allocations['Date'] = self.time_span['Date']
        daily_allocations.set_index('Date', inplace=True)  # Ensure Date is the index
        return daily_allocations.reset_index()  # Reset index for Altair

    def __string__(self):
        return self.element
    
    def get_stocks(self):
        return list(self.stockData.keys())
    
    def get_price_on_day(self, stock, day):
        df = self.stockData[stock].get_adj_close().copy()
        df['Date'] = pd.to_datetime(df['Date'])
        day = pd.to_datetime(day)
        match = df[df['Date'] == day]
        if not match.empty:
            return float(match['Adj Close'].iloc[0])
        else:
            raise ValueError(f"❌ Date {day.date()} not found in stock '{stock}' data.")


    def evaluate(self):
        money = self.start_value
        old_day = self.time_span['Date'].iloc[0]
        array_money = [money]
        
        for i, day in enumerate(self.time_span['Date'][1:], start=1):
            new_money = 0
            for stock in self.stockData:
                money_stock_owned = self.pf[stock][i-1] * money
                old_stock_price = self.get_price_on_day(stock, old_day)
                stock_owned = money_stock_owned / old_stock_price
                current_stock_price = self.get_price_on_day(stock, day)
                new_money += current_stock_price * stock_owned
            money = new_money
            array_money.append(money)
            old_day = day
        return array_money

    def _generate_random_pf(self, stocks):
        pf = pd.DataFrame({stock: [randint(1, 100) for _ in range(self.num_day)] for stock in stocks})
        return pf.div(pf.sum(axis=1), axis=0)

    def evaluate_pf(self, pf):
        money = self.start_value
        old_day = self.time_span['Date'].iloc[0]
        array_money = [money]
        
        for i, day in enumerate(self.time_span['Date'][1:], start=1):
            new_money = 0
            for stock in self.stockData:
                money_stock_owned = pf[stock][i-1] * money
                old_stock_price = self.get_price_on_day(stock, old_day)
                stock_owned = money_stock_owned / old_stock_price
                current_stock_price = self.get_price_on_day(stock, day)
                new_money += current_stock_price * stock_owned
            money = new_money
            array_money.append(money)
            old_day = day
        return money

class OptimizationAlgorithm:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def optimize(self):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

class GeneticAlgorithm(OptimizationAlgorithm):

    def __init__(self, portfolio, k_genetic_mutation, k_modified_elephant, k_crossover):
        super().__init__(portfolio)
        self.k_genetic_mutation = k_genetic_mutation
        self.k_modified_elephant = k_modified_elephant
        self.k_crossover = k_crossover

    def evolve_population(self, population):
        new_population = []
        for _ in range(len(population) // 2):
            if random.random() < self.k_crossover:
                parent1, parent2 = random.sample(population, 2)
                child = parent1.copy()
                for day in child.index:
                    if random.random() < 0.5:
                        child.loc[day] = parent2.loc[day]
                new_population.append(child)
        
        for elephant in population:
            if random.random() < self.k_modified_elephant:
                baby_elephant = elephant.copy()
                for day in baby_elephant.index:
                    if random.random() < self.k_genetic_mutation:
                        baby_elephant.loc[day] = [random.randint(1, 100) for _ in baby_elephant.columns]
                        baby_elephant.loc[day] /= baby_elephant.loc[day].sum()
                new_population.append(baby_elephant)
        return new_population

    def filter_population(self, scored_population, num_best):
        max_value = max(ele[1] for ele in scored_population)
        min_value = min(ele[1] for ele in scored_population)
        
        # Normalizzazione dei punteggi
        normalized_scores = [
            (idx, ele[0], ele[1], ele[2]/100 * ((ele[1] * ele[1]) - min_value) / (max_value - min_value))
            if max_value != min_value else (idx, ele[0], ele[1], 1)
            for idx, ele in enumerate(scored_population)
        ]
        
        # Ordinamento per fitness decrescente
        normalized_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Selezione sicura dei primi 3
        selected_population = [(ele[1], ele[2]) for ele in normalized_scores[:3]]
        selected_indices = set(ele[0] for ele in normalized_scores[:3])
        
        # Selezione probabilistica per il resto
        while len(selected_population) < num_best and len(selected_indices) < len(normalized_scores):
            selected = random.choices(normalized_scores, weights=[ele[3] for ele in normalized_scores], k=1)[0]
            if selected[0] not in selected_indices:
                selected_population.append((selected[1], selected[2]))
                selected_indices.add(selected[0])
        
        return [(elephant, fitness, randint(50, 100)) for elephant, fitness in selected_population]

    def optimize(self, num_initial_population, loop_num):
        population = [self.portfolio._generate_random_pf(self.portfolio.stockData.keys()) for _ in range(num_initial_population)]
        scored_population = [[elephant, self.portfolio.evaluate_pf(elephant), randint(50, 100)] for elephant in population]

        for _ in range(loop_num):
            new_population = self.evolve_population(population)
            scored_new_population = [[elephant, self.portfolio.evaluate_pf(elephant), randint(50, 100)] for elephant in new_population]
            scored_population.extend(scored_new_population)
            scored_population = self.filter_population(scored_population, num_initial_population)
            population = [elephant for elephant, _, _ in scored_population]
            print("Best portfolio value:", max(scored_population, key=lambda x: x[1])[1])
            print("=====================================")
            self.portfolio.pf = max(scored_population, key=lambda x: x[1])[0]
            yield max(scored_population, key=lambda x: x[1])[0]

class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, portfolio, initial_temp=100, cooling_rate=0.95, max_iter=1000):
        self.portfolio = portfolio
        self.T = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.best_pf = portfolio.pf.copy()
        evaluate_result = portfolio.evaluate()
        self.best_score = evaluate_result[-1] if isinstance(evaluate_result, list) else evaluate_result.iloc[-1]['Money']


    def step(self):
        new_pf = self._get_new_portfolio()
        new_pf = new_pf.div(new_pf.sum(axis=1), axis=0)
        self.portfolio.pf = new_pf
        evaluate_result = self.portfolio.evaluate()
        new_score = evaluate_result[-1] if isinstance(evaluate_result, list) else evaluate_result.iloc[-1]['Money']

        delta = new_score - self.best_score

        if delta > 0 or np.random.rand() < np.exp(delta / self.T):
            self.best_pf = new_pf.copy()
            self.best_score = new_score

        self.T *= self.cooling_rate
    

    def _get_new_portfolio(self):
        new_pf = self.portfolio.pf.copy()
        
        # Randomly select two different stocks
        stocks = self.portfolio.get_stocks()
        stock1, stock2 = np.random.choice(stocks, 2, replace=False)

        # Randomly choose a transfer amount (max 5% of stock1's value)
        delta = np.random.uniform(0.01, min(0.05, new_pf[stock1].min()))

        # Ensure stock1 does not go below 0 and stock2 does not exceed 1
        if new_pf[stock1].min() - delta >= 0 and new_pf[stock2].max() + delta <= 1:
            new_pf[stock1] -= delta
            new_pf[stock2] += delta

        # ✅ **Normalize allocations to ensure they sum to 1**
        new_pf = new_pf.div(new_pf.sum(axis=1), axis=0)

        return new_pf
    
    def optimize(self):
        current_pf = self.portfolio.pf.copy()
        best_pf = current_pf.copy()
        best_score = self.portfolio.evaluate()[-1]

        for _ in range(self.max_iter):
            new_pf = self._get_new_portfolio()
            self.portfolio.pf = new_pf
            new_score = self.portfolio.evaluate()[-1]

            delta = new_score - best_score
            if delta > 0 or np.random.rand() < np.exp(delta / self.T):
                best_pf = new_pf.copy()
                best_score = new_score

            self.T *= self.cooling_rate
            if self.T < 0.01:
                break

        self.portfolio.pf = best_pf
        return best_pf, best_score

class HillClimbing(OptimizationAlgorithm):
    def __init__(self, portfolio, iterations=1000, step_size=0.3):
        super().__init__(portfolio)
        self.iterations = iterations
        self.step_size = step_size

    def modify_allocation(self, allocation):
        new_allocation = allocation.copy()
        stocks = list(new_allocation.columns)
        
        # Generate a vector with a random block set to 1 and others set to 0
        block_size = np.random.randint(1, len(stocks) // 2 + 1)  # Block size between 1 and half the number of stocks
        block_stocks = np.random.choice(stocks, block_size, replace=False)
        block_vector = np.zeros(len(stocks))
        for stock in block_stocks:
            block_vector[stocks.index(stock)] = 1  # Set the block to 1
        
        # Apply a random change to the selected block
        change = np.random.uniform(-0.3, 0.3)  # Change allocation by ±0.3
        for stock in block_stocks:
            new_allocation[stock] = (new_allocation[stock] + change).clip(0, 1)  # Clip values to [0,1]
        
        # Normalize allocations to ensure they sum to 1
        new_allocation = new_allocation.div(new_allocation.sum(axis=1), axis=0)
        return new_allocation

    def optimize(self):
        current_pf = self.portfolio.pf.copy()
        best_pf = current_pf.copy()
        best_eval = self.portfolio.evaluate_pf(current_pf)

        for _ in range(self.iterations):
            new_pf = self.modify_allocation(current_pf)
            new_eval = self.portfolio.evaluate_pf(new_pf)

            if new_eval > best_eval:
                best_pf = new_pf.copy()
                best_eval = new_eval
                current_pf = new_pf

        self.portfolio.pf = best_pf
        return best_pf, best_eval 

class TabuSearch(OptimizationAlgorithm):

    def __init__(self, portfolio, iterations=100, tabu_size=50):
        super().__init__(portfolio)
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.tabu_list = []

    def modify_allocation(self, allocation):
        new_allocation = allocation.copy()
        stocks = list(new_allocation.columns)
        
        # Generate a vector with a random block set to 1 and others set to 0
        block_size = np.random.randint(1, len(stocks) // 2 + 1)  # Block size between 1 and half the number of stocks
        block_stocks = np.random.choice(stocks, block_size, replace=False)
        block_vector = np.zeros(len(stocks))
        print(block_vector)
        for stock in block_stocks:
            block_vector[stocks.index(stock)] = 1  # Set the block to 1


        vector = np.zeros(self.portfolio.num_day)
        length = random.randint(1, self.portfolio.num_day // 2)
        start = np.random.randint(0, self.portfolio.num_day - length + 1)  # Punto iniziale casuale
        vector[start:start + length] = 1

        
        # Apply a random change to the selected block
        change = np.random.uniform(-0.3, 0.3)  # Change allocation by ±0.3
        change *= vector
        for stock in block_stocks:
            new_allocation[stock] = (new_allocation[stock] + change).clip(0, 1)  # Clip values to [0,1]

        
        # Normalize allocations to ensure they sum to 1
        new_allocation = new_allocation.div(new_allocation.sum(axis=1), axis=0)
        return new_allocation


    def optimize(self):
        current_pf = self.portfolio.pf.copy()
        best_pf = current_pf.copy()
        best_eval = self.portfolio.evaluate_pf(current_pf)

        print(self.iterations)
        
        for _ in range(self.iterations):
            new_pf = self.modify_allocation(current_pf)
            new_eval = self.portfolio.evaluate_pf(new_pf)
            pf_signature = hash(tuple(np.round(new_pf.values.flatten(), 3)))

            # Eğer tabu listesinde değilse ve daha iyiyse
            if pf_signature not in self.tabu_list and new_eval > best_eval:
                best_pf = new_pf.copy()
                best_eval = new_eval
                current_pf = new_pf
                self.tabu_list.append(str(new_pf.values.tolist()))
                if len(self.tabu_list) > self.tabu_size:
                    self.tabu_list.pop(0)  # Maintain tabu list size

                yield best_pf
                
        
    
if __name__ == '__main__':
    # Write here code for running without Stremlit UI
    pass





