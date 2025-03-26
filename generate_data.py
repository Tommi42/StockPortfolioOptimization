import pandas as pd
import numpy as np

# Genera date lavorative dal 2016-01-01 al 2016-07-28
dates = pd.bdate_range(start="2016-01-01", end="2016-07-28")

# Funzione per generare dati sintetici di stock
def generate_stock_data(base_price, trend, volatility, crash_date=None):
    prices = []
    price = base_price
    for date in dates:
        # Se c'è un crash a metà periodo
        if crash_date and date >= crash_date:
            price *= 0.95  # Perdita improvvisa
        else:
            price *= (1 + trend + np.random.uniform(-volatility, volatility))

        open_price = price * np.random.uniform(0.98, 1.02)
        high_price = price * np.random.uniform(1.00, 1.05)
        low_price = price * np.random.uniform(0.95, 1.00)
        close_price = price

        prices.append([date, open_price, high_price, low_price, close_price, close_price, np.random.randint(100000, 500000)])
    
    return prices

# Generazione dati per 3 stock
stock_A = generate_stock_data(100, trend=0.002, volatility=0.005)  # Crescita costante
stock_B = generate_stock_data(100, trend=0.002, volatility=0.005, crash_date=pd.Timestamp("2016-04-01"))  # Cresce e poi crolla
stock_C = generate_stock_data(50, trend=0.0, volatility=0.02)  # Molto volatile

# Creazione DataFrame
columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_A = pd.DataFrame(stock_A, columns=columns)
df_B = pd.DataFrame(stock_B, columns=columns)
df_C = pd.DataFrame(stock_C, columns=columns)

# Salvataggio su file CSV
df_A.to_csv("ES1.csv", index=False)
df_B.to_csv("ES2.csv", index=False)
df_C.to_csv("ES3.csv", index=False)
