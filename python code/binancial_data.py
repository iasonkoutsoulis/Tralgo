# this fetches the financial data
import numpy as np
import pyarrow.feather as feather
import yfinance as yf

tics = 'msft aapl goog tsla'

df = yf.download(tics, interval = "1wk", start='2009-10-01')
feather.write_feather(df, 'E:/Tralgo/data/financial_container.csv')

df = feather.read_feather('E:/Tralgo/data/financial_container.csv')
df = df[['Adj Close']].dropna()
df.index = df.index.strftime('%Y-%m') + '-' + np.where(df.index.day>=15, 'B2', 'B1')
df = df.groupby(df.index).mean()
for col in df:
    df[col[1] + '_lag'] = df[col].shift(1)
    df[col[1] + '_indicator'] = df[col] - df[col[1] + '_lag']
    df[col[1] + '_indicator'] = np.where(df[col[1] + '_indicator'] >=0, 1, 0)

# td = TDClient(apikey="89a252537e204b4682c6855d0d7f4f89")
# ts = td.time_series(
#     symbol="SP500TR",
#     interval="1week",
#     outputsize=728
# )
# df = ts.as_pandas()[["close"]]
# df.index = df.index.strftime('%Y-%m') + '-' + np.where(df.index.day>=15, 'B2', 'B1')

# # df = pd.read_csv('data/hicp_raw.csv')
# # df.rename(columns=lambda x: re.sub(r'^(\w+).*', r'\1', x))

# to do:
# replace twelvedata with yfinance for this, as we don't need a live feed, only established data