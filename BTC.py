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
import scipy.stats

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
ts = pd.DataFrame(btc['Close']).set_index(rng)

missing = []
for dti in btc.index:
    if dti in rng:
        print('{} is in rng: {}'.format(dti, dti in rng))
        missing.append(dti)

set(rng).difference(set(missing))
## missing dates from 24-06-2016 to 23-07-2016

present = []
for ud in np.unique(btc.index):
    if btc.ix[btc.index == ud].shape[0] > 1:
        present.append(ud)

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
                         ### N.B.: dates from 2013

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

#### Tuning GMM model:
covs = [ 'spherical', 'tied', 'diag', 'full']
nc = [4,9,12]
for c in covs:
    for n in nc:
        mGMM = mixture.GMM(n_components = n, covariance_type = c).fit(dataset)
        print("""GMM model with {} covariance,
              and {} components:
              aic = {}
              bic = {}
              likelihood = {}""".format(c, n, mGMM.aic(dataset), mGMM.bic(dataset),
                mGMM.score_samples(dataset)[0].sum()))        
############## best: 12 components and full covariance

                
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

###############################################################################

close = btc['Close']
dclose = np.diff(close)

plt.figure()
plt.plot(dclose)

ret = []
for i in range(close.size - 1):
    ret.append((close.ix[i+1] - close.ix[i])/close.ix[i])
    
returns = np.array(ret)

plt.figure()
plt.plot(returns)
plt.axhline(y = scipy.stats.mstats.mquantiles(np.array(returns), prob = 0.975))
plt.axhline(y = scipy.stats.mstats.mquantiles(np.array(returns), prob = 0.025))

plt.figure()
plt.plot(statsmodels.api.tsa.acf(returns))

returns = pd.Series(returns)
plt.figure()
plotting.autocorrelation_plot(returns)
plt.figure()
plotting.lag_plot(returns)

plt.figure()
returns.hist()

plt.figure()
plt.plot(statsmodels.api.tsa.periodogram(np.array(returns)))

per = statsmodels.api.tsa.periodogram(np.array(returns))
per[per > 0.01].size

plt.figure()
plt.plot(np.array(returns), color = 'red')
plt.plot(fourierExtrapolation(np.array(returns), 0, 154))

plt.figure()
plt.plot(np.array(close))

### zoom
plt.figure()
plt.plot(np.array(close)[1240:1300], color = 'magenta')
plt.figure()
plt.plot(np.array(returns)[1600:1700], color = 'magenta', marker = 'o')

###############################################################################
def get_ROI(ts):
    rets = []
    for i in range(ts.size - 1):
        rets.append((ts.ix[i+1] - ts.ix[i])/ts.ix[i])
    return np.array(rets)
###############################################################################
def detect_trend_at_t(ts, start, end):
    rets = get_ROI(ts)
    rets= rets[start:end]
    trend = []
    for t in range(1, rets.size, 1):
        rets2 = rets[:t]
        trend.append((rets2[rets2 >= 0].size)/rets2.size)
    return trend
###############################################################################

trend = detect_trend_at_t(close, 1600, 1700)    

###############################################################################
def trend_is_changing(ts, window):
    rets = get_ROI(ts)
    diffs = np.diff(ts)
    ct = []
    for i in range(2*window, diffs.size - window, 1):
        jump = diffs[i]
        q = scipy.stats.mstats.mquantiles(diffs[(i-window):i], prob = [0.025, 0.975])
        if jump < q[0] or jump > q[1]:
            bef_rets = rets[(i-window):i]
            aft_rets = rets[i:(i+window)]
            num_pos = bef_rets[bef_rets > 0].size/bef_rets.size
            num_pos_ = aft_rets[aft_rets > 0].size/aft_rets.size
            if num_pos > 0.5 and num_pos_ < 0.5:
                ct.append((i, 'negative change'))
            elif num_pos < 0.5 and num_pos_ > 0.5:
                ct.append((i, 'positive change'))
            else:
                pass
    return ct
###############################################################################

E = trend_is_changing(close, 50)    

points = []
for i in E:
    points.append(i[0])

plt.figure()
plt.plot(np.array(close))
plt.scatter(np.array(points), np.array(close.ix[points]), color = 'black', marker = 'o')

qnts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
for q in qnts:
    print(scipy.stats.mstats.mquantiles(dclose, prob = q))

ixs = []
for i in range(close.size - 1):
    if close.ix[i+1] - close.ix[i] >   16.74506:
        ixs.append(i)

plt.figure()
plt.plot(np.array(close))
plt.scatter(np.array(ixs), np.array(close.ix[ixs]), color = 'black', marker = 'o')
        
plt.figure()
plt.plot(np.linspace(0,2280,2280), np.repeat(0, 2280))
plt.scatter(np.array(ixs), np.repeat(0, len(ixs)), color = 'black', marker = 'o')

#### on returns:
qnts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
for q in qnts:
    print(scipy.stats.mstats.mquantiles(returns, prob = q))

ixs = []
for i in range(close.size - 1):
    if abs(close.ix[i+1] - close.ix[i])/(close.ix[i]) > 0.09451776:
        ixs.append(i)

plt.figure()
plt.plot(np.array(close), color = 'grey')
plt.scatter(np.array(ixs), np.array(close.ix[ixs]), color = 'black', marker = 'o')
        
plt.figure()
plt.plot(np.linspace(0,2280,2280), np.repeat(0, 2280))
plt.scatter(np.array(ixs), np.repeat(0, len(ixs)), color = 'black', marker = 'o')


###############################################################################
def template_pattern(ts, start, end):
    vec = ts[start:end]
    vec_st = (vec - np.mean(vec))/np.std(vec)
    app = np.polyfit(np.linspace(0, vec_st.size, vec_st.size), vec_st, 3)
    return app
###############################################################################
def find_pattern(ts, tmp, window):
    deriv = np.array([3*tmp[0], 2*tmp[1], tmp[2]])
    dpoly = np.poly1d(deriv)
    fp = []
    errs = []
    for i in range(ts.size - window):
        loc = ts[i:(i+window)]
        loc_st = (loc - np.mean(loc))/np.std(loc)
        loctmp = np.polyfit(np.linspace(0,loc_st.size,loc_st.size),loc_st,3)
        dloctmp = np.array([3*loctmp[0], 2*loctmp[1], loctmp[2]])
        locpoly = np.poly1d(dloctmp)
        errs.append(np.sqrt(np.mean((locpoly - dpoly)**2)))
        if np.sqrt(np.mean((locpoly - dpoly)**2)) <= 1e-2:
            print('pattern found')            
            fp.append(i)
    print('{} patterns found'.format(len(fp)))
    return fp, errs
###############################################################################
def approximating_polynomial(x, yhat):
    xx = 0
    for y in yhat[:(yhat.size-1)]:
        deg = yhat.size - (yhat.tolist().index(y) + 1)
        xx += (x**deg) * y
    return xx + yhat[-1]
############################################################################### 

tmp = template_pattern(close, 1240, 1300)    
Ps, Es = find_pattern(close, tmp, 100)    

plt.figure()
plt.plot(np.array(close), color = 'purple')
plt.scatter(np.array(Ps), np.array(close.ix[Ps]), color = 'black', marker = 'o')

gr = np.array(close)[1240:1300]
stgr = (gr - np.mean(gr))/(np.std(gr))
plt.figure()
plt.plot(stgr, color = 'magenta')
plt.plot(np.linspace(0, 60, 60), approximating_polynomial(np.linspace(0, 60, 60),tmp))

plt.figure()
plt.hist(np.array(Es))

ws = [20,40,60,80,100,200]
for w in ws:
    tmp = template_pattern(close, 1240, 1300)    
    Ps, Es = find_pattern(close, tmp, w)    
    print(scipy.stats.skew(np.array(Es)))
    print(scipy.stats.kurtosis(np.array(Es)))
    print('#################################################################')
    
##############################
mc = []
vc = []
for i in range(1, close.size, 1):
    mc.append(np.mean(close.ix[:i]))    
    vc.append(np.var(close.ix[:i]))    
    
plt.figure()
plt.plot(np.array(mc))    
plt.figure()
plt.plot(np.sqrt(np.array(vc)))

#############################

statsmodels.api.tsa.stattools.arma_order_select_ic(np.array(returns),max_ar=5, max_ma=5, ic=['aic', 'bic'], trend='nc')

#method = [‘css-mle’,’mle’,’css’]
mod = statsmodels.api.tsa.ARMA(np.array(returns), order=(4,2)).fit(merthod = 'css',full_output = True)
mod.forecast(steps = 8)
print(mod.params)
print(mod.resid)
phat = mod.predict(0, 2276)

plt.figure()
plt.plot(returns[:500])
plt.plot(phat[:500], color = 'red')

res = returns - phat
np.mean(res)
np.std(res)

sam = statsmodels.api.tsa.arma_generate_sample(ar = 1, ma = 3, nsample = 250)

statsmodels.api.tsa.adfuller(returns) ### rejects the null hypothesis that there is a unit root (i.e. proces is 
                                      ### NON stationary) => returns are stationary.