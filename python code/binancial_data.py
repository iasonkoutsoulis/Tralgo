# this fetches the financial data
import numpy as np
import pyarrow.feather as feather
import yfinance as yf

tics = 'msft aapl goog tsla'

df = yf.download(tics, interval = "1wk", start='2009-10-01')
df = df[['Adj Close']].dropna()
df.columns = df.columns.droplevel()
df.index = df.index.strftime('%Y-%m') + '-' + np.where(df.index.day>=15, 'B2', 'B1')
df = df.groupby(df.index).mean()
for col in df:
    df[col + '_lag'] = df[col].shift(1)
    df[col + '_indicator'] = df[col] - df[col + '_lag']
    df[col + '_indicator'] = np.where(df[col + '_indicator'] >=0, 1, 0)
    df[col + '_future_indicator'] = df[col + '_indicator'].shift(-1)

feather.write_feather(df, 'E:/Tralgo/data/financial_container.csv')