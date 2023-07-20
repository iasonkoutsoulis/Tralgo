# this fetches the financial data
import re
import pandas as pd
from twelvedata import TDClient

td = TDClient(apikey="89a252537e204b4682c6855d0d7f4f89")
ts = td.time_series(
    symbol="EUR/USD",
    interval="1month",
    outputsize=168,
    timezone="America/New_York",
)
df = ts.as_pandas()

# df = pd.read_csv('data/hicp_raw.csv')
# df.rename(columns=lambda x: re.sub(r'^(\w+).*', r'\1', x))

