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
        self.time_span = pd.DataFrame({"Date": self.stockData[stocks[0]].get_adj_close()['Date']})

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

    def evaluate(self):
        money = self.start_value
        old_day = self.time_span['Date'].iloc[0]
        array_money = [money]
        
        for i, day in enumerate(self.time_span['Date'][1:], start=1):
            new_money = 0
            for stock in self.stockData:
                money_stock_owned = self.pf[stock][i-1] * money
                old_stock_price = float(self.stockData[stock].get_adj_close().loc[self.stockData[stock].get_adj_close()['Date'] == old_day]['Adj Close'].iloc[0])
                stock_owned = money_stock_owned / old_stock_price
                current_stock_price = float(self.stockData[stock].get_adj_close().loc[self.stockData[stock].get_adj_close()['Date'] == day]['Adj Close'].iloc[0])
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
                old_stock_price = float(self.stockData[stock].get_adj_close().loc[self.stockData[stock].get_adj_close()['Date'] == old_day]['Adj Close'].iloc[0])
                stock_owned = money_stock_owned / old_stock_price
                current_stock_price = float(self.stockData[stock].get_adj_close().loc[self.stockData[stock].get_adj_close()['Date'] == day]['Adj Close'].iloc[0])
                new_money += current_stock_price * stock_owned
            money = new_money
            array_money.append(money)
            old_day = day
        return money

class GeneticAlgorithm:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def evolve_population(self, population, k_genetic_mutation, k_modified_elephant, k_crossover):
        new_population = []
        for _ in range(len(population) // 2):
            if random.random() < k_crossover:
                parent1, parent2 = random.sample(population, 2)
                child = parent1.copy()
                for day in child.index:
                    if random.random() < 0.5:
                        child.loc[day] = parent2.loc[day]
                new_population.append(child)
        
        for elephant in population:
            if random.random() < k_modified_elephant:
                baby_elephant = elephant.copy()
                for day in baby_elephant.index:
                    if random.random() < k_genetic_mutation:
                        baby_elephant.loc[day] = [random.randint(1, 100) for _ in baby_elephant.columns]
                        baby_elephant.loc[day] /= baby_elephant.loc[day].sum()
                new_population.append(baby_elephant)
        return new_population

    def filter_population(self, scored_population, num_best):
        max_value = max(ele[1] for ele in scored_population)
        min_value = min(ele[1] for ele in scored_population)
        normalized_scores = [
            (idx, ele[0], ele[1], ele[2]/100 * (ele[1] - min_value) / (max_value - min_value))
            if max_value != min_value else (idx, ele[0], ele[1], 1)
            for idx, ele in enumerate(scored_population)
        ]
        normalized_scores.sort(key=lambda x: x[2], reverse=True)
        selected_population = []
        selected_indices = set()
        while len(selected_population) < num_best and len(selected_indices) < len(normalized_scores):
            selected = random.choices(normalized_scores, weights=[ele[3] for ele in normalized_scores], k=1)[0]
            if selected[0] not in selected_indices:
                selected_population.append((selected[1], selected[2]))
                selected_indices.add(selected[0])
        return [(elephant, fitness, randint(50, 100)) for elephant, fitness in selected_population]

    def run(self, num_initial_population, k_genetic_mutation, k_modified_elephant, k_crossover, loop_num):
        population = [self.portfolio._generate_random_pf(self.portfolio.stockData.keys()) for _ in range(num_initial_population)]
        scored_population = [[elephant, self.portfolio.evaluate_pf(elephant), randint(50, 100)] for elephant in population]
        
        for _ in range(loop_num):
            new_population = self.evolve_population(population, k_genetic_mutation, k_modified_elephant, k_crossover)
            scored_new_population = [[elephant, self.portfolio.evaluate_pf(elephant), randint(50, 100)] for elephant in new_population]
            scored_population.extend(scored_new_population)
            scored_population = self.filter_population(scored_population, num_initial_population)
            population = [elephant for elephant, _, _ in scored_population]
            print("Best portfolio value:", max(scored_population, key=lambda x: x[1])[1])
            print("=====================================")
            self.portfolio.pf = max(scored_population, key=lambda x: x[1])[0]
            yield max(scored_population, key=lambda x: x[1])[0]
    
class SimulatedAnnealing:

    def __init__(self, portfolio, initial_temp=100, cooling_rate=0.99, max_iter=1000):
        self.portfolio = portfolio  # Portfolio nesnesi
        self.T = initial_temp  # Başlangıç sıcaklığı
        self.cooling_rate = cooling_rate  # Soğuma oranı
        self.max_iter = max_iter  # Maksimum iterasyon sayısı

    def _get_new_portfolio(self):
        """
        Generates a new portfolio with small adjustments while ensuring valid percentages.
        """
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
        """
        Simulated Annealing algorithm for portfolio optimization.
        """
        current_pf = self.portfolio.pf.copy()
        best_pf = current_pf.copy()
        best_score = self.portfolio.evaluate().iloc[-1]['Money']

        for i in range(self.max_iter):
            new_pf = self._get_new_portfolio()
            
            # **Normalize to avoid values outside [0,1]**
            new_pf = new_pf.div(new_pf.sum(axis=1), axis=0)
            
            self.portfolio.pf = new_pf  # Update temporary allocation
            new_score = self.portfolio.evaluate().iloc[-1]['Money']

            delta = new_score - best_score

            if delta > 0:
                best_pf = new_pf.copy()
                best_score = new_score
            else:
                prob = np.exp(delta / self.T)
                if np.random.rand() < prob:
                    best_pf = new_pf.copy()
                    best_score = new_score

            self.T *= self.cooling_rate  # Cool down temperature
            if self.T < 0.01:
                break

        # ✅ **Fix: Ensure SA updates Portfolio.pf correctly**
        self.portfolio.pf = best_pf.div(best_pf.sum(axis=1), axis=0)  # Normalize again

        return best_pf, best_score

class HillClimbing:
    def __init__(self, portfolio, iterations=1000, step_size=0.3):
        self.portfolio = portfolio  # Portfolio object
        self.iterations = iterations  # Number of optimization steps
        self.step_size = step_size  # Max % change in stock allocation

    def optimize(self):
        """Optimize portfolio using Hill Climbing."""
        current_pf = self.portfolio.pf.copy()  # Start with the current allocation
        current_eval = self.portfolio.evaluate_pf(current_pf)  # Evaluate initial money
        best_pf, best_eval = current_pf, current_eval

        for _ in range(self.iterations):
            new_pf = self.modify_allocation(current_pf)  # Generate new allocation
            new_eval = self.portfolio.evaluate_pf(new_pf)  # Evaluate new portfolio

            if new_eval > best_eval:  # Keep the new allocation only if it's better
                best_pf, best_eval = new_pf, new_eval
                current_pf = new_pf  # Continue optimizing from the new best

        self.portfolio.pf = best_pf  # Update portfolio with best allocation
        return best_pf, best_eval

    def modify_allocation(self, allocation):
        """Modify allocation slightly."""
        new_allocation = allocation.copy()
        stock = np.random.choice(list(new_allocation.columns))  # Pick a random stock
        change = np.random.uniform(-self.step_size, self.step_size)  # Change allocation by ± step_size
        
        new_allocation[stock] = (new_allocation[stock] + change).clip(0, 1)  # Clip values to [0,1]
        new_allocation = new_allocation.div(new_allocation.sum(axis=1), axis=0)  # Normalize
        return new_allocation
    
if __name__ == '__main__':
    p_random = Portfolio(1000, ["ALL", "A2M", "AGL"], '2016-01-01', '2017-01-01')
    print(p_random.pf)
    print(p_random.evaluate())
    p_sa = Portfolio(1000, ["ALL", "A2M", "AGL"], '2016-01-01', '2017-01-01')

    ga = GeneticAlgorithm(p_random)
    temp_best = ga.run(30, 0.3, 0.9, 0.5, 20)
    print("Finito!")

    print(p_random.pf)
    print(p_random.evaluate())


    """
    # Rastgele portföy değeri
    random_value = p_random.evaluate()

    # Simulated Annealing portföy değeri
    sa = SimulatedAnnealing(p_sa)
    best_pf, best_score = sa.optimize()

    sa_value = p_sa.evaluate()

    # Grafik karşılaştırması
    plt.figure(figsize=(10, 6))
    plt.plot(random_value['Money'], label='Random Portfolio', linestyle='--')
    plt.plot(sa_value['Money'], label='Simulated Annealing (Optimized)', linewidth=2)
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Sonuçların Yazdırılması
    print("\n--- Random Portfolio ---")
    print(f"Final Value: {random_value.iloc[-1]['Money']:.2f}")
    print(p_random.pf)

    print("\n--- Simulated Annealing (Optimized Portfolio) ---")
    print(f"Final Value: {best_score:.2f}")
    print(best_pf)"
    """
