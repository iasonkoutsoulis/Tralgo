# this fetches the financial data
from twelvedata import TDClient
import yfinance as yf
import numpy as np

tics = 'msft aapl goog tsla'

df = yf.download(tics, interval = "1wk", start='2009-10-01')
df = df[['Adj Close']]
df.index = df.index.strftime('%Y-%m') + '-' + np.where(df.index.day>=15, 'B2', 'B1')
df = df.groupby(df.index).mean()


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