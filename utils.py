import csv
import datetime
import pandas as pd
from random import randint
from typing import List 
import matplotlib.pyplot as plt
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

        if self.bool_filtered_data == False:
            self.bool_filtered_data = True
            
            start_date = self.start_date
            end_date = self.end_date

            if start_date != None:
                start_date = pd.to_datetime(start_date)
            else:
                start_date = pd.to_datetime('1900-01-01')
            
            if end_date != None:
                end_date = pd.to_datetime(end_date)
            else:
                end_date = pd.to_datetime('2100-01-01')
            
            try:
                self.filtered_data = self.data[(pd.to_datetime(self.data['Date']) > start_date) & (pd.to_datetime(self.data['Date']) < end_date)][['Date','Adj Close']]
            except Exception as e:
                print("Error with timespan. Probably not all stock selected have data for the selected timespan")
                print(f"{e}")

        return self.filtered_data

class Portfolio():

    def __init__(self, start_value: int, stocks: List[str], start_date, end_date):
        self.start_value = start_value
        self.stockData = {stock: DataReader(f"./archive/{stock}.csv", start_date, end_date) for stock in stocks}
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
        old_i = 0
       
        array_money = [money]

        for (i, day)  in enumerate(self.time_span['Date'][1:]):
            new_money = 0
            i += 1
            for stock in self.stockData:

                # Money of stock owned from the day [-1]
                money_stock_owned = self.pf[stock][old_i] * money # Here money are still the Old ammount

                old_stock_price = float(self.stockData[stock].get_adj_close().loc[self.stockData[stock].get_adj_close()['Date'] == old_day]['Adj Close'].iloc[0])

                stock_owned = money_stock_owned / old_stock_price
                
                # Stock price at day [i] from previus day
                current_stock_price = float(self.stockData[stock].get_adj_close().loc[self.stockData[stock].get_adj_close()['Date'] == day]['Adj Close'].iloc[0])

                new_money += (current_stock_price * stock_owned)

            money = new_money
            array_money.append(money)
            old_i = i
            old_day = day

        return pd.DataFrame({'Money': array_money})

            # relative_temp_stock = float(self.pf.loc[self.pf['Date'] == day][stock].iloc[0]) # Stock % that day
            # money_temp_stock = relative_temp_stock * money # Actual ammount of money invested in that stock
            # temp_stock = money_temp_stock * # Actual ammount of stock owned ad day [current]

            # temp_price = float(dd.loc[dd['Date'] == day]["Adj Close"].iloc[0]) 
            # money += (temp_amount - temp_pick) * temp_price
            # temp_stock_old = temp_pick

        return money

    def get_stocks(self):
        return [stock for stock in self.stockData]

    def get_portfolio(self):
        return self.pf

    def get_portfolio_sampled(self, N):
        pf = self.pf
        if len(pf) > N:
            df_sampled = pf.iloc[::len(pf)//N][:N]  # Passo regolare e limitazione a N righe
        else:
            df_sampled = pf

        return df_sampled

    def hill_climbin(self):
        pass



import numpy as np

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




if __name__ == '__main__':
    p_random = Portfolio(1000, ["ALL", "A2M", "AGL"], '2016-01-01', '2017-01-01')
    p_sa = Portfolio(1000, ["ALL", "A2M", "AGL"], '2016-01-01', '2017-01-01')
    
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
    print(best_pf)



   
    
    