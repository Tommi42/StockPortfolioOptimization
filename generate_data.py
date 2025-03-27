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

# Generazione dati per ES4 (crollo di 5 giorni e poi ripresa)
def generate_crash_recovery_data(base_price, trend, volatility, crash_start, crash_end):
    prices = []
    price = base_price
    for date in dates:
        if crash_start <= date <= crash_end:
            price *= 0.9  # Perdita improvvisa durante il crollo
        else:
            price *= (1 + trend + np.random.uniform(-volatility, volatility))
        open_price = price * np.random.uniform(0.98, 1.02)
        high_price = price * np.random.uniform(1.00, 1.05)
        low_price = price * np.random.uniform(0.95, 1.00)
        close_price = price
        prices.append([date, open_price, high_price, low_price, close_price, close_price, np.random.randint(100000, 500000)])
    return prices

# Generazione dati per ES5 (picco di 5 giorni e poi ritorno alla normalità)
def generate_peak_recovery_data(base_price, trend, volatility, peak_start, peak_end):
    prices = []
    price = base_price
    for date in dates:
        if peak_start <= date <= peak_end:
            price *= 1.1  # Picco improvviso durante il periodo
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

# Generazione dati per ES4 e ES5
crash_start = pd.Timestamp("2016-03-01")
crash_end = pd.Timestamp("2016-03-05")
peak_start = pd.Timestamp("2016-05-01")
peak_end = pd.Timestamp("2016-05-05")

stock_D = generate_crash_recovery_data(80, trend=0.001, volatility=0.01, crash_start=crash_start, crash_end=crash_end)
stock_E = generate_peak_recovery_data(120, trend=0.001, volatility=0.01, peak_start=peak_start, peak_end=peak_end)

# Creazione DataFrame
columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_A = pd.DataFrame(stock_A, columns=columns)
df_B = pd.DataFrame(stock_B, columns=columns)
df_C = pd.DataFrame(stock_C, columns=columns)
df_D = pd.DataFrame(stock_D, columns=columns)
df_E = pd.DataFrame(stock_E, columns=columns)

# Salvataggio su file CSV
df_A.to_csv("ES1.csv", index=False)
df_B.to_csv("ES2.csv", index=False)
df_C.to_csv("ES3.csv", index=False)
df_D.to_csv("ES4.csv", index=False)
df_E.to_csv("ES5.csv", index=False)
