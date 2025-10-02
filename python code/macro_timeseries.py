import pandas as pd
import numpy as np
import pandas_datareader as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt

# datafetch
start = '1947-01-01'
end = None

fred_q = ['GDPC1', 'GDPDEF', 'PCEC', 'FPI', 'CE16OV', 'DFF', 'CNP16OV', 'EMRATIO', 'PRS85006023', 'PRS85006101']
df = pdr.DataReader(fred_q, 'fred', start, end)

# real gross domestic product -SA annual
# gdp implicit price deflator -SA
# personal consumption expenditures -SA annual
# fixed private investment -SA annual
# civilian employment 16 and over -index 1992:3 = 1 -SA
# FFR -avg of daily -percent
# pop lvl 16 and over: -nSA
# labor force status: civilian noninstitutional population 16 over: SA, in thou -same index
# nonfarm business all persons avg weekly hours duration -SA -1992 id
# same but wages (hrly compensiation duration) -SA

# datawork
df_q = df.resample('Q').mean()

for col in df_q.columns:
    s = df_q[col].dropna()
    if s.empty:
        continue

    plt.figure()             # ensure a separate figure
    s.plot()                 # matplotlib (no seaborn, no colors specified)
    plt.title(col)
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.tight_layout()
    # Show (if running interactively) and save
    # Comment out savefig if you don't want files
    plt.show()