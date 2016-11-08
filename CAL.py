# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:18:48 2016

@author: d_floriello

CAL analysis
"""

import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy.misc import derivative
import seaborn as sns
from scipy.stats import levy
from sklearn import linear_model
from sklearn import  neighbors
import scipy

cal = pd.read_excel('CAL.xlsx', sheetname = 'valori CAL')
#cal = cal.fillna(0)

##################
#### example with levy ####
l = levy.fit(cal['AS16'])
sampei = levy.rvs(loc = l[0], scale = l[1], size = 100)


#################

monthwise = OrderedDict()

mesi = ['gen', 'feb', 'mar', 'apr', 'mag', 'giu', 'lug', 'ago', 'set', 'ott', 'nov', 'dic']

for y in range(10,18,1):
    varn = 'AS'+str(y)
    res = np.repeat(0, 12)
    for m in range(len(mesi)):
        res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].mean()
    monthwise[varn] = res
    
mw = pd.DataFrame.from_dict(monthwise).set_index([mesi])
mw['AS17'].ix['ott'] = (39.7+40.1)/2

mw.plot()

mw.mean(axis=1).plot()

##############################################

normalized_monthwise = OrderedDict()

for y in range(10,18,1):
    varn = 'AS'+str(y)    
    res = np.repeat(0, 12)    
    if y < 17:        
        for m in range(len(mesi)):
            res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].mean()
        res2 = (res - np.mean(res))/np.std(res)
        normalized_monthwise[varn] = res2
    else:
        for m in range(11):
            if m < 10:
                res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].mean()
            else:
                res[m] = cal[varn].ix[cal[cal.columns[8]] == mesi[m]].sum()/2
        res2 = (res - np.mean(res))/np.std(res)
        normalized_monthwise[varn] = res2
        
normalized_monthwise

nmw = pd.DataFrame.from_dict(normalized_monthwise).set_index([mesi])
nmw['AS17'].ix['nov'] = nmw['AS17'].ix['dic'] = 0 

#############################################

diffs = OrderedDict()

for y in range(10,18,1):
    varn = 'AS'+str(y)
    diffs[varn] = np.diff(cal[varn])
    plt.figure() 
    plt.plot(np.diff(cal[varn]))
    plt.suptitle(varn)
    
df = pd.DataFrame.from_dict(diffs)

############################################
def find_peaks_return(var, df, d, bVerbose = True):
    mu = df[var].mean()
    sigma = df[var].std()
    freq = np.count_nonzero( (df[var]-mu)/sigma > d )/df.shape[0]
    print('frequency of points greater than {} = {}'.format(d, freq))
    diff_values = np.diff(df[var].ix[np.abs((df[var]-mu)/sigma) > d])
    if bVerbose:
            print('values differences between peaks: {}'.format(diff_values))        
    num_contr = np.diff(df[var].ix[np.abs((df[var]-mu)/sigma) > d].index)
    if bVerbose: 
        print('distance between peaks:'.format(num_contr))
    return diff_values, num_contr
###########################################

dv, nc = find_peaks_return('AS10', df, 1, False)

means = []
for y in range(10,18,1):
    varn = 'AS'+str(y)
    dv, nc2 = find_peaks_return(varn, df, 1, False)
    print('media num contrattazioni di picco:'.format(np.nanmean(nc2)))
    means.append(np.mean(nc2))

###########################################
### how far are the max and min from the mean? and from the initial value? 

stats = OrderedDict()

stats['mins'] = cal.min()
stats['maxs'] = cal.max()
stats['means'] = cal.mean()
stats['stds'] = cal.std()
stats['starts'] = cal.ix[0]
stats['ends'] = cal.ix[cal.shape[0] - 1]
stats['skews'] = cal.skew()

stats = pd.DataFrame.from_dict(stats)

######################################################
def std_distance(x, y, s):
    print('{} is at {} sigmas from {}'.format(x, abs(x - y)/s, y))
    return abs(x - y)/s
#####################################################

std_distance(stats['mins'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])
std_distance(stats['means'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])
std_distance(stats['starts'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])
std_distance(stats['ends'].ix['AS16'], stats['maxs'].ix['AS16'], stats['stds'].ix['AS16'])

#####################################################
def rolling_dist(ts):
    dist = []
    for i,x in enumerate(ts[1:]):
        pmean = np.mean(ts[:i])
        psig = np.std(ts[:i])
        dist.append(std_distance(x, pmean, psig))
    return np.array(dist)
######################################################
        
rd = rolling_dist(cal['AS16'])

plt.figure()
plt.plot(rd, marker='o')

######################################################
def plot_mean_graphs(ts):
    #ts = cal['AS16']
    cum_mu = []
    for i,x in enumerate(ts):
        cum_mu.append(np.mean(ts[:i]))
    cum_mu = np.array(cum_mu)
    plt.figure()
    plt.plot(cum_mu, marker = 'o', color = 'magenta')
    plt.figure()
    plt.plot(np.diff(cum_mu), marker = 'o', color = 'lime')
    return cum_mu, np.diff(cum_mu)
######################################################
mu,diff = plot_mean_graphs(cal['AS17'])

#####################################################

f_mean = interp1d(np.linspace(-1,  cal.shape[0], cal.shape[0]), cal['AS16'])

der = []
for x in np.linspace(5, cal.shape[0], 2*cal.shape[0]):
    print(x)
    der.append(derivative(f_mean, x))

plt.figure()
plt.plot(np.array(der))

####################################################
def cumulative_diffs(ts, bVerbose = False):
    M = np.max(ts)
    v = [] ### dist from cumulative mean
    curr = [] ### dist from current value
    st = []
    for i,x in enumerate(ts[1:]):
        if bVerbose:
            M = x
        if np.isnan((M - x)/np.std(ts[:i])) or np.isinf((M - x)/np.std(ts[:i])):
            pass
        else:
            curr.append((M - x)/np.std(ts[:i]))
        if np.isnan((M - np.mean(ts[:i]))/np.std(ts[:i])) or np.isinf((M - np.mean(ts[:i]))/np.std(ts[:i])):
            pass
        else:
            v.append((M - np.mean(ts[:i]))/np.std(ts[:i])) 
        if np.isnan((M - ts[0])/np.std(ts[:i])) or np.isinf((M - ts[0])/np.std(ts[:i])):
            pass
        else:
            st.append((M - ts[0])/np.std(ts[:i]))
            
    mark = [ts.tolist().index(np.max(ts)),ts.tolist().index(np.min(ts))] 
        
    plt.figure()
    plt.plot(np.array(v), '-gD', markevery = mark)
    plt.title('max vs cumulative mean')        
    plt.figure()
    plt.plot(np.array(curr), '-gD', markevery = mark)
    plt.title('max vs current value')        
    plt.figure()
    plt.plot(np.array(st), '-gD', markevery = mark)
    plt.title('max vs first value')        
        
    return v, curr, st
###################################################
dist_mu, current, dist_st = cumulative_diffs(cal['AS16'], True) 
dist_mu, current, dist_st = cumulative_diffs(cal['AS15'], True) 
dist_mu, current, dist_st = cumulative_diffs(cal['AS14'], True) 


plt.figure()
plt.plot(dist_mu[2:])

plt.figure()
plt.plot(current[2:])
plt.figure()
plt.plot(dist_st[2:])

###################################################
def H1_distance(ts1, ts2):
    return np.mean((ts1 - ts2)**2) + np.mean((np.diff(ts1) - np.diff(ts2))**2)
###################################################
    
sim = OrderedDict()
    
for cn1 in cal.columns[:7]:
    dist = []
    for cn2 in cal.columns[:7]:
        print('{} with {} = {}'.format(cn1, cn2, H1_distance(cal[cn1], cal[cn2])))
        dist.append(H1_distance(cal[cn1], cal[cn2]))
    sim[cn1] = dist
    
sim = pd.DataFrame.from_dict(sim).set_index([['AS10','AS11','AS12','AS13','AS14','AS15','AS16']])

sns.heatmap(sim)

log_cal = OrderedDict()

for cn in cal.columns[:8]:
    log_cal[cn] = np.diff(np.log(cal[cn]))
    
log_cal = pd.DataFrame.from_dict(log_cal)

############################################################################
def cumulative_maxmin(ts):
    M = []
    m = []
    for i in range(ts.size):
        M.append(np.max(ts[:i]))
        m.append(np.min(ts[:i]))
    plt.figure()
    plt.plot(np.array(M))
    plt.title('cumulative max')
    plt.figure()
    plt.plot(np.array(m))
    plt.title('cumulative min')
    return M, m
############################################################################
def sell(x, y, hold, threshold = 0):
    if hold: ### hold means I have something bought at a given price
        if y - x > threshold:
            return y - x
        else:
            return 0
    else:
        return 0
############################################################################  
def buy(last, y, free, threshold = 0):
    if free:
        if y < abs(last - threshold):
            return -y
        else:
            return 0
    else:
        return 0
############################################################################            
def simulate_strategy(ts, threshold = 0):
    x = last = ts[0]
    hold = True
    free = False
    portfolio = left = 0
    for y in ts[1:]:
        if hold:
            portfolio += sell(x, y, hold, threshold)
            hold = False
            last = y
            free = True
        elif free:
            portfolio += buy(last, y, free, threshold)
            free = False
            x = last
            hold = True
        else:
            left += 1
    print('days with no operations: {}'.format(left))
    return portfolio
###########################################################################
###########################################################################
###########################################################################
class Portfolio:
    
    def __init__(self, value, hold, free, threshold_buy, threshold_sell):
        self.value = value
        self.hold = hold ## long position
        self.free = free ## short position
        self.last_held_value = 0
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.trading_days = 0
        
    def get_portfolio_value(self):
        return self.value
        
    def get_position_long(self):
        return self.hold

    def get_position_short(self):
        return self.free        
        
    def get_thr_buy(self):
        return self.threshold_buy        

    def get_thr_sell(self):
        return self.threshold_sell        

    def set_thr_buy(self, x):
        self.threshold_buy = x

    def set_thr_sell(self, x):
        self.threshold_sell = x

    def buy(self, price):
        if self.get_position_short():
            if price < self.get_thr_buy():
                print('buying {}'.format(price))
                self.value -= price
                self.last_held_value = price
                self.set_thr_buy(price)
                self.set_thr_sell(price)
                self.free = False
                self.hold = True
            else:
                print('too expensive')
        else:
            print("I'm long already")
            
    def sell(self, gain):
        if self.get_position_long():
            if gain > self.get_thr_sell():
                print('selling {}'.format(gain))
                self.value += gain
                self.hold = False
                self.free = True
            else:
                print("it's not worthy selling it")
        else:
            print("I'm short already")
        
    def simulate_strategy_naive(self, process, bVerbose = False):
        start = process[0]
        self.last_held_value = start
        self.set_thr_sell(start)
        self.set_thr_buy(start)
        self.hold = True
        self.free = False
        self.value -= start
        self.trading_days += 1
        last_portfolio_value = self.value
        for i,x in enumerate(process[1:]):
            if bVerbose:            
                print('step {}'.format(i))
                print('portfolio value BEFORE operations: {}'.format(self.get_portfolio_value()))
            ### check if I can sell and if it's worthy:
            self.sell(x)
            ### check if I can buy and it's worthy:
            self.buy(x)
            if bVerbose:            
                print('portfolio value AFTER operations: {}'.format(self.get_portfolio_value()))            
            if self.value != last_portfolio_value:
                self.trading_days += 1
                last_portfolio_value = self.value
        print('final portfolio value = {}'.format(self.get_portfolio_value()))
        return self.get_portfolio_value()


###############################################################################
###############################################################################
###############################################################################
def Cuscore_Statistics(ts):
    CS = []
    betas = []
    cs_incr = 0
    epsilon = 1e-6
    X = np.array(list(range(8))).reshape(-1, 1)
    init = linear_model.LinearRegression(fit_intercept = True).fit(X, ts[:8])
    beta_hat = init.coef_[0]
    for i in range(9, ts.size - 3, 1):
        print('step {}'.format(i))
        Xnew = np.array(list(range(i))).reshape(-1, 1)
        regr = linear_model.LinearRegression(fit_intercept = True).fit(Xnew, ts[:i])
        beta_new = regr.coef_[0]
        
        X_last3 = np.array(list(range(i-3, i, 1))).reshape(-1, 1)
        regr_last3 = linear_model.LinearRegression(fit_intercept = True).fit(X_last3, ts[(i-3):i])
        beta_last3 = regr_last3.coef_[0]
        
        X_last = np.array(list(range(i-8, i+3, 1))).reshape(-1, 1)
        regr_last = linear_model.LinearRegression(fit_intercept = True).fit(X_last, ts[(i-8):(i+3)])
        beta_last = regr_last.coef_[0]
        
        if beta_new * beta_hat > epsilon and beta_new * beta_last > epsilon: 
        ### trend doesn't change
            print('trend does not change')
            beta_hat = beta_new  
            cs_incr += (ts[i] - beta_hat * i) * i 
            CS.append(cs_incr)            
        
        elif beta_hat * beta_new < epsilon and beta_hat * beta_last > epsilon: 
        ### it could change, but it's actually a local trend 
            print('trend does not change')            
            beta_hat = beta_new
            cs_incr += (ts[i] - beta_hat * i) * i 
            CS.append(cs_incr)
            
        elif beta_hat * beta_new < epsilon and beta_hat * beta_last < epsilon:
        ### trend is changing
            print('trend does change')
            beta_hat = beta_last
            cs_incr += (ts[i] - beta_hat * i) * i 
            CS.append(cs_incr)

        elif beta_new * beta_hat > epsilon and beta_new * beta_last < epsilon: 
            if beta_hat * beta_last3 > epsilon:
                beta_hat = beta_new  
                cs_incr += (ts[i] - beta_hat * i) * i 
                CS.append(cs_incr)
            else:
                beta_hat = beta_last
                cs_incr += (ts[i] - beta_hat * i) * i 
                CS.append(cs_incr)
        
        else:
            print('case not considered')
        betas.append(beta_hat)
        
    zoo = [1 if beta >= 0 else 0 for beta in betas]

    indexes = []    
    indexes.append(0)
    for i in range(len(zoo)-1):
        if zoo[i] == 1 and zoo[i+1] == 0:
            indexes.append(i)
        elif zoo[i] == 0 and zoo[i+1] == 1:
            indexes.append(i)
        else:
            pass
    
    cs_corrected = []
    for i in range(len(indexes)-1):
        for j in range(indexes[i], indexes[i+1],1):
            cs_corrected.append((ts[j] - betas[indexes[i]] * j) * j)
        
    return cs_incr, cs_corrected, betas, zoo    
#    return cs_incr, cs_corrected, [betas[x] for x in indexes]#betas[indexes]
###############################################################################
###############################################################################
ts = cal['AS17'].ix[cal['AS17'] > 0] ### to remove the nan's
err, beta, x = detect_trends_man(ts)        

yt = []
for i in range(len(x)-1):
    intercept = 0
    if i == 0:
        intercept = np.mean(ts[0:x[1]])
    else:
        intercept = yt[-1][-1]
    yt.append(beta[i] * range(x[i], x[i+1], 1) + intercept)
 
yt.append(beta[-1] * range(x[-1], ts.size, 1) + ts.ix[x[-1]])

yt = np.concatenate(yt).ravel()
yt = pd.Series(yt)

plt.figure()
plt.plot(ts)
plt.figure()
plt.plot(yt)








c_stat, seq, coffs, zero = Cuscore_Statistics(ts.ix[ts.index.month >= 8])                

st = np.array(ts.ix[ts.index.month >= 8])

X_ = np.array(list(range(st.size))).reshape(-1, 1)
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X_, st)

line_y_ransac = model_ransac.predict(X_)
plt.plot(st)
plt.plot(X_, line_y_ransac, '-b', label='RANSAC regressor')

for i in range(2,st.size,1):
    print(i)
    X_ = np.array(list(range(st[:i].size))).reshape(-1, 1)
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(X_, st[:i])
    line_y_ransac = model_ransac.predict(X_)
    plt.figure()
    plt.plot(st[:i])
    plt.plot(X_, line_y_ransac, '-b', label='RANSAC regressor')
#    plt.figure()
#    plt.plot(st[:i])
#    plt.plot(beta*np.linspace(0, 20, num = 50) + 40.5, '-gD', markevery=i)

plt.figure()
plt.plot(np.array(seq))
plt.figure()
plt.plot(np.array(coffs))



n_neighbors = 5
X = np.linspace(1, ts.size, num = ts.size)[:, np.newaxis]
### distance better than uniform
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_ = knn.fit(X, ts.ravel()).predict(X)
plt.figure()
plt.plot(X, y_, c='g', label='prediction')


###############################################################################
cal = pd.read_excel('cal_17.xlsx')
cal = cal.set_index(cal['Date'])
cal = cal['CAL17']

plt.figure()
cal.plot()
cal.hist(bins = 20)

mmin = []
mmax = []
for i in range(cal.shape[0]):
    mmin.append(cal.ix[:i].min())
    mmax.append(cal.ix[:i].max())

plt.figure()
plt.plot(np.array(cal))
plt.plot(np.array(mmin))
plt.plot(np.array(mmax))

dcal = np.diff(cal)
plt.figure()
plt.plot(dcal)
plt.axhline(scipy.stats.mstats.mquantiles(dcal, prob = 0.025))
plt.axhline(scipy.stats.mstats.mquantiles(dcal, prob = 0.975))
plt.plot(np.diff(mmin))
plt.plot(np.diff(mmax))

###############################################################################
def cumulative_quantiles(ts):
    cqu = []
    cql = []
    for i in range(10, ts.size, 1):
        cqu.append(scipy.stats.mstats.mquantiles(ts[:i], prob = 0.975)[0])
        cql.append(scipy.stats.mstats.mquantiles(ts[:i], prob = 0.025)[0])
    return np.array(cqu), np.array(cql)
###############################################################################

cq1, cq2 = cumulative_quantiles(dcal)
        
plt.figure()
plt.figure()
plt.plot(dcal)
plt.axhline(scipy.stats.mstats.mquantiles(dcal, prob = 0.025))
plt.axhline(scipy.stats.mstats.mquantiles(dcal, prob = 0.975))
plt.plot(cq1)
plt.plot(cq2)

np.where(dcal <= scipy.stats.mstats.mquantiles(dcal, prob = 0.025)[0])[0].size
np.where(dcal >= scipy.stats.mstats.mquantiles(dcal, prob = 0.975)[0])[0].size

roi = []
lambda_t = []
for i in range(cal.shape[0] - 1):
    r_o_i = (cal.ix[i+1] - cal.ix[i])/cal.ix[i]
    roi.append(r_o_i)
    lambda_t.append(cal.ix[i+1]/cal.ix[i])

roi = np.array(roi)
cqr1, cqr2 = cumulative_quantiles(roi)

lambda_t = np.array(lambda_t)

plt.figure()
plt.plot(roi)
plt.plot(cqr1)
plt.plot(cqr2)

plt.figure()
plt.plot(lambda_t)

qrl = scipy.stats.mstats.mquantiles(roi, prob = 0.025)[0]
qru = scipy.stats.mstats.mquantiles(roi, prob = 0.975)[0]

np.sum(roi[roi >= qru] - roi[roi <= qrl])

np.where(roi[10:] >= cqr1)[0].size
np.where(roi[10:] <= cqr2)[0].size

roi10 = roi[10:]
np.sum(roi10[roi10 >= cqr1][:22] - roi10[roi10 <= cqr2]) 

