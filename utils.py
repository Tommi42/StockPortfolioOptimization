import csv
import datetime
import pandas as pd
from random import randint
from typing import List 

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

        pf = pd.DataFrame({ stock: [randint(1,100) for i in range(len(self.stockData[stocks[0]].get_adj_close()['Date']))] for stock in stocks})
    
        temp_sum = pf.sum(axis=1)

        self.pf = pf.div(temp_sum, axis=0)

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

if __name__ == '__main__':
    p = Portfolio(1000, ["ALL", "A2M", "AGL"], '2016-01-01', '2017-01-01')
    print(p.evaluate())
    print(p.pf)


   
    
    