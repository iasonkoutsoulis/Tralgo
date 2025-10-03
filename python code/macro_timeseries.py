import pandas as pd
import numpy as np
import pandas_datareader as pdr
import statsmodels as sm
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
df_q.index = df_q.index.to_period('Q')

# practically we index with 2017Q3 because that's what PRS85006023 uses

df_q['CE_idx'] = 100 * df_q['CE16OV'] / df_q['CE16OV'].loc['2017Q3']
df_q['CNP_idx'] = df_q['CNP16OV'] / df_q['CNP16OV'].loc['2017Q3']

# index hourly wage
s = df_q['PRS85006101']
g = (1 + s/100).groupby(df_q.index.quarter)
chain = g.cumprod()
df_q['PRS85006101_idx'] = 100 * chain / chain.loc['2017Q3']

# print(df_q['PRS85006101_idx'].to_string())
# checking that values match between here and usmodel_data.xsl

df_q['consumption'] = np.log( (df_q['PCEC'] / df_q['GDPDEF']) / df_q['CNP_idx'] ) * 100 # note: GDPDEF is lower by around 4
df_q['investment'] = np.log( (df_q['FPI'] / df_q['GDPDEF']) / df_q['CNP_idx'] ) * 100
df_q['output'] = np.log( df_q['GDPC1'] / df_q['CNP_idx'] ) * 100
df_q['hours'] = np.log( (df_q['PRS85006023']*df_q['CE_idx'] / 100) / df_q['CNP_idx'] ) * 100
df_q['inflation'] = ( np.log(df_q['GDPDEF']) - np.log(df_q['GDPDEF'].shift(1))  )* 100
df_q['real_wage'] = np.log( df_q['PRS85006101_idx'] / df_q['GDPDEF'] ) * 100
df_q['interest_rate'] = df_q['DFF'] / 4

# final set of vars in the usmodel_data.xls
df_q['dc'] = df_q['consumption'] - df_q['consumption'].shift(1)
df_q['dinve'] = df_q['investment'] - df_q['investment'].shift(1)
df_q['dy'] = df_q['output'] - df_q['output'].shift(1)
df_q['labobs'] = df_q['hours'] - df_q['hours'].mean()
df_q['pinfobs'] = df_q['inflation']
df_q['dw'] = df_q['real_wage'] - df_q['real_wage'].shift(1)
df_q['robs'] = df_q['interest_rate']

Y = df_q.loc[:, 'dc':'robs'].dropna()

## VAR
# examine autocorrelation

x_order = ['dy', 'dc', 'dinve', 'dw', 'labobs', 'pinfobs', 'robs']
X = Y[x_order]
X.index = X.index.to_series().astype(str)

sm.graphics.tsaplots.plot_acf(X['labobs'].dropna(), lags=30, title="dy ACF")
sm.graphics.tsaplots.plot_pacf(X['labobs'].dropna(), lags=30, title="dy PACF")
plt.show()

model = sm.tsa.api.VAR(X)
results = model.fit(4, trend='c')
results.summary()
results.plot()
plt.show()

irfs = results.irf(20)
fig = irfs.plot(impulse='robs', orth=False)
plt.show()