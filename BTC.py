# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:08:34 2016

@author: d_floriello

BTC analysis
"""

import pandas as pd
from pandas.tools import plotting
import statsmodels.api
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

btc = pd.read_csv('C:/Users/d_floriello/Documents/bitcoin (2).csv')

btc = btc.set_index(pd.DatetimeIndex(btc['Date']))
btc.head()
btc = btc[['Open', 'High', 'Low', 'Close']]

plt.figure()
btc['Close'].plot()

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(btc['Close'])))
plt.title('BTC periodogram')

btc_3 = btc.ix[btc.index.year >= 2013]
btc_3.plot()

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram( (np.array(btc_3['Close']) - np.mean(np.array(btc_3['Close'])))/np.std(np.array(btc_3['Close'])) ))
plt.title('BTC periodogram (from 2013)')

btc_3M = btc_3.resample('M').mean()

plt.figure()
btc_3M.plot()

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram( (np.array(btc_3M['Close']) - np.mean(np.array(btc_3M['Close'])))/np.std(np.array(btc_3M['Close'])) ))
plt.title('MBTC periodogram (from 2013)')

###############################################################################
def fourierExtrapolation(x, n_predict, n_harmonics = 0):
    x = np.array(x)
    n = x.size
    if n_harmonics == 0:
        n_harm = 100                     # number of harmonics in model
    else:
        n_harm = n_harmonics
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t        
###############################################################################
plt.figure()
plt.plot(np.array(btc_3M))
plt.plot(fourierExtrapolation(np.array(btc_3M).ravel(), 0, 2), color = 'red')    

dl = np.diff(np.log(btc['Close']))
plt.figure()
plt.plot(dl)
d = np.diff(btc['Close'])
plt.figure()
plt.plot(d)

rng = pd.date_range(btc.index[0], btc.index[-1], freq = 'D')
ts = pd.DataFrame(btc['Close']).set_index()

missing = []
for dti in btc.index:
    if dti in rng:
        print('{} is in rng: {}'.format(dti, dti in rng))
        missing.append(dti)

set(rng).difference(set(missing))
## missing dates from 24-06-2016 to 23-07-2016

dec = statsmodels.api.tsa.seasonal_decompose(pd.Series(btc['Close']), freq = 28)

plt.figure()
dec.plot()

plt.figure()
plt.plot(np.array(dec.seasonal[0:28]))

change = pd.read_excel('C:/Users/d_floriello/Documents/change.xlsx')
change = change.set_index(change['Date'])
change = change['change']
plt.figure()
change.plot()

cd = set(change.index).intersection(btc.index)
ibtc = set(btc.index).intersection(cd)

tsbtc = pd.Series(btc['Close'].ix[list(ibtc)])
tsc = pd.Series(change.ix[list(cd)])

plt.figure()
tsbtc.plot()
plt.axhline(y = 200)
plt.axhline(y = 500)
plt.figure()
tsc.plot()

tsbtc.corr(tsc)

plt.figure()
plotting.lag_plot(tsbtc) ### surprising!!! I think the reticular squared structure puts in evidence the 
                         ### particular pattern tat I've noticed.

data_bit = []
for i in range(tsbtc.size-1):
    xy = np.array([tsbtc.ix[i],tsbtc.ix[i+1]])
    data_bit.append(xy)

dataset = np.array(data_bit)

from sklearn import mixture

model = mixture.GMM(n_components = 9, covariance_type = 'full').fit(dataset)
x = np.linspace(0, 1200, num = 1200)
X, Y = np.meshgrid(x, x)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model.score_samples(XX)[0]
Z = Z.reshape(X.shape)

plt.figure()
CS = plt.contour(X, Y, Z, levels=np.logspace(0, 100, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(dataset[:, 0], dataset[:, 1], .8)
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()

###############################################################################
def conditional_distribution(df, xs, xe):
    edf1 = df.ix[xs <= df[df.columns[0]].values]
    edf = edf1.ix[edf1[edf1.columns[0]].values < xe]
    #plt.figure()
    #edf.hist()
    return edf
###############################################################################
def simple_MarkovMatrix(df, xs, xe):
    # http://stats.stackexchange.com/questions/14360/estimating-markov-chain-probabilities
# http://stats.stackexchange.com/questions/41145/simple-way-to-algorithmically-identify-a-spike-in-recorded-errors    
    diz = OrderedDict()
    edf = conditional_distribution(df, xs, xe)
    m0 = edf[0].min()
    M0 = edf[0].max()
    #dr0 = M0 - m0
    #step0 = dr0/10
    m1 = df[1].min()
    M1 = df[1].max()
    #dr1 = M1 - m1
    #step1 = dr1/10
    lin0 = np.linspace(m0, M0, 10)
    lin1 = np.linspace(m1, M1, 10)
    for i in range(lin0.size - 1):
        cedf = conditional_distribution(edf, lin0[i], lin0[i+1])
        sizer = cedf.shape[0]
        vec = []
        for j in range(lin1.size - 1):
            cedf1 = cedf.ix[lin1[j] <= cedf[cedf.columns[1]].values]
            cedf2 = cedf1.ix[cedf1[cedf1.columns[1]].values < lin1[j + 1]]
            vec.append(cedf2.shape[0]/sizer)
        diz[str(lin0[i])] = vec
    df = pd.DataFrame.from_dict(diz, orient = 'index')
    df.columns = [[str(lin1[j]) for j in range(lin1.size-1)]]
    return df
###############################################################################

tsbtc = pd.Series(btc['Close'])

data_bit = []
for i in range(tsbtc.size-1):
    xy = np.array([tsbtc.ix[i],tsbtc.ix[i+1]])
    data_bit.append(xy)

dataset = np.array(data_bit)


df = pd.DataFrame(dataset)

t1 = conditional_distribution(df, 200, 500)
df1 = simple_MarkovMatrix(df, 200, 500)
df2 = simple_MarkovMatrix(df, df[0].min(), df[0].max())