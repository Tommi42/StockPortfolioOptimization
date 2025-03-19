import csv
import datetime
import pandas as pd
from random import randint
from typing import List 


class DataReader():

    def __init__(self, data_csv):

        self.name = data_csv.strip('.csv').split('/')[1]

        self.data = pd.read_csv(data_csv)
                
    def __str__(self):
        return f"--- Data of {self.name} stock ---\n{str(self.data)}"
        
    def adj_close(self, start_date, end_date: str):
        if start_date != None:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = pd.to_datetime('1900-01-01')
        
        if end_date != None:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = pd.to_datetime('2100-01-01')
        
        filtered_data = self.data[(pd.to_datetime(self.data['Date']) > start_date) & (pd.to_datetime(self.data['Date']) < end_date)][['Date','Adj Close']]
        return filtered_data

class Portfolio():

    def __init__(self, start_value: int, stocks: List[str]):
        self.start_value = start_value
        self.element = pf = pd.DataFrame({'Date': []})

    def __string__(self):
        return self.element

    def evaluate():
        pass

if __name__ == '__main__':
    a = DataReader('archive/ALL.csv')
    b = DataReader('archive/A2M.csv')

    d = a.adj_close('2016-01-01', '2018-01-01')['Date']
    dd = a.adj_close('2016-01-01', '2018-01-01')

    pf = pd.DataFrame({'Date': d,
                       'ALL': [randint(0,1) for i in range(len(d))],
                       'A2M': [randint(0,1) for i in range(len(d))],
                       })
    temp_amount = 0

    money = 1000
    
    for day in d:
        temp_pick = float(pf.loc[pf['Date'] == day]['ALL'].iloc[0])
        temp_price = float(dd.loc[dd['Date'] == day]["Adj Close"].iloc[0])
        money += (temp_amount - temp_pick) * temp_price
        temp_amount = temp_pick

    print(money)

    
    