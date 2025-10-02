# this fetches the financial data and tests for stationarity
import numpy as np
import pyarrow.feather as feather
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

tics = 'msft aapl goog tsla'

df = yf.download(tics, interval = "1wk", start='2008-10-01')
df = df[['Close']].dropna()
df.columns = df.columns.droplevel()
df.index = df.index.strftime('%Y-%m') + '-' + np.where(df.index.day>=15, 'B2', 'B1')
df = df.groupby(df.index).mean()
for col in df:
    df[col + '_dif'] = df[col].diff()
    df[col + '_indicator'] = np.where(df[col + '_dif'] >=0, 1, 0)
    df[col + '_future_indicator'] = df[col + '_indicator'].shift(-1)

adfuller(df['GOOG_dif'].dropna())

feather.write_feather(df, '/data/financial_container.csv')