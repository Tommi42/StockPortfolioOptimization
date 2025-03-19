import csv
import datetime
from pypfopt import risk_models
from pypfopt import expected_returns
import pandas as pd


class DataReader():

    def __init__(self, data_csv):

        self.name = data_csv.strip('.csv').split('/')[1]

        with open(data_csv, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            self.data = []
            i = 0
            for row in spamreader:
                if i == 0:
                    i += 1
                    continue
                else:
                    line = []
                    d = row[0].split('-')
                    line.append(datetime.datetime(int(d[0]), int(d[1]), int(d[2])))
                    for i in range(1, len(row)):
                        try:
                            line.append(float(row[i]))
                        except:
                            line.append(0)


                    self.data.append(line)


        self.p_data = pd.read_csv(data_csv)
                
    def __str__(self):
        str = ''
        for row in self.data:
            str += f'Date: {row[0]} | Open: {round(row[1],1)} | High: {round(row[2],1)} | Low: {round(row[3],1)} | Close: {round(row[4],1)} | Adj Close: {round(row[5],1)} | Volume: {round(row[6])}\n'
        return str

    def avarage_close(self):
        sum = 0
        for row in self.data:
            sum += row[4]

        return sum / len(self.data)
    
    def avarage_volume(self):
        sum = 0
        for row in self.data:
            sum += row[6]

        return sum / len(self.data)
    
    def data_btime(self, t: datetime.datetime):
        filtered_data = [row for row in self.data if row[0] < t]
        return filtered_data
    

    def esxpected_return_daily(self, t: datetime.datetime):
        sum = 0
        historical_data = self.data_btime(t)
        for day in historical_data:
            sum += (day[4] - day[1])

        return (sum / len(historical_data))
    
    def esxpected_return_monthly(self, t: datetime.datetime):
        sum = 0
        historical_data = self.data_btime(t)
        last_day = historical_data[0]
        i = 0

        for day in historical_data:
            if day[0].year != last_day[0].year or day[0].month != last_day[0].month:
                sum += last_day[4]
                sum -= day[1]
                i += 1


            last_day = day

        return (sum / i)
    
    def esxpected_return_yearly(self, t: datetime.datetime):
        sum = 0
        historical_data = self.data_btime(t)
        last_day = historical_data[0]
        i = 0

        for day in historical_data:
            if day[0].year != last_day[0].year:
                sum += last_day[4]
                sum -= day[1]
                i += 1


            last_day = day

        return (sum / i)

    
    def adj_close(self, t: datetime.datetime):
        return self.p_data['Adj Close']


if __name__ == '__main__':
    d = DataReader('archive/ALL.csv')
    print(d.avarage_close())
    print(d.avarage_volume())
    # Example usage of data_btime
    df = d.adj_close(datetime.datetime(2015, 1, 1))
    mu = expected_returns.mean_historical_return(df)
    print(mu['Adj Close'])