import csv
import datetime


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
    

if __name__ == '__main__':
    d = DataReader('archive/ALL.csv')
    print(d.avarage_close())
    print(d.avarage_volume())
    print(d)