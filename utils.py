import csv
import datetime
import pandas as pd
from random import randint
import random
from typing import List

class DataReader:
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
        self.pf = self._generate_random_pf(stocks)
    
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
        return None

if __name__ == '__main__':
    portfolio = Portfolio(1000, ["ALL", "A2M", "AGL"], '2016-01-01', '2017-01-01')
    ga = GeneticAlgorithm(portfolio)
    ga.run(30, 0.3, 0.9, 0.5, 20)
    print("Finito!")
