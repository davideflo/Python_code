# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:20:27 2017

@author: utente

Fit Levy processes for Trading Simulation
"""

import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools import plotting
from sklearn.linear_model import LinearRegression
import statsmodels.api
import scipy
import math
import numpy
import random
import decimal
import scipy.linalg
import numpy.random as nrand

####################################################################################################
####################################################################################################
class ModelParameters:
    """
    Encapsulates model parameters
    """

    def __init__(self,
                 all_s0, all_time, all_delta, all_sigma, gbm_mu,
                 jumps_lamda=0.0, jumps_sigma=0.0, jumps_mu=0.0,
                 cir_a=0.0, cir_mu=0.0, all_r0=0.0, cir_rho=0.0,
                 ou_a=0.0, ou_mu=0.0,
                 heston_a=0.0, heston_mu=0.0, heston_vol0=0.0):
        # This is the starting asset value
        self.all_s0 = all_s0
        # This is the amount of time to simulate for
        self.all_time = all_time
        # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
        self.all_delta = all_delta
        # This is the volatility of the stochastic processes
        self.all_sigma = all_sigma
        # This is the annual drift factor for geometric brownian motion
        self.gbm_mu = gbm_mu
        # This is the probability of a jump happening at each point in time
        self.lamda = jumps_lamda
        # This is the volatility of the jump size
        self.jumps_sigma = jumps_sigma
        # This is the average jump size
        self.jumps_mu = jumps_mu
        # This is the rate of mean reversion for Cox Ingersoll Ross
        self.cir_a = cir_a
        # This is the long run average interest rate for Cox Ingersoll Ross
        self.cir_mu = cir_mu
        # This is the starting interest rate value
        self.all_r0 = all_r0
        # This is the correlation between the wiener processes of the Heston model
        self.cir_rho = cir_rho
        # This is the rate of mean reversion for Ornstein Uhlenbeck
        self.ou_a = ou_a
        # This is the long run average interest rate for Ornstein Uhlenbeck
        self.ou_mu = ou_mu
        # This is the rate of mean reversion for volatility in the Heston model
        self.heston_a = heston_a
        # This is the long run average volatility for the Heston model
        self.heston_mu = heston_mu
        # This is the starting volatility value for the Heston model
        self.heston_vol0 = heston_vol0
####################################################################################################
####################################################################################################
def convert_to_returns(log_returns):
    """
    This method exponentiates a sequence of log returns to get daily returns.
    :param log_returns: the log returns to exponentiated
    :return: the exponentiated returns
    """
    return numpy.exp(log_returns)
####################################################################################################
def convert_to_prices(param, log_returns):
    """
    This method converts a sequence of log returns into normal returns (exponentiation) and then computes a price
    sequence given a starting price, param.all_s0.
    :param param: the model parameters object
    :param log_returns: the log returns to exponentiated
    :return:
    """
    returns = convert_to_returns(log_returns)
    # A sequence of prices starting with param.all_s0
    price_sequence = [param.all_s0]
    for i in range(1, len(returns)):
        # Add the price at t-1 * return at t
        price_sequence.append(price_sequence[i - 1] * returns[i - 1])
    return numpy.array(price_sequence)
####################################################################################################    
def plot_stochastic_processes(processes, title):
    """
    This method plots a list of stochastic processes with a specified title
    :return: plots the graph of the two
    """
    plt.style.use(['bmh'])
    fig, ax = plt.subplots(1)
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('Time, t')
    ax.set_ylabel('Simulated Asset Price')
    x_axis = numpy.arange(0, len(processes[0]), 1)
    for i in range(len(processes)):
        plt.plot(x_axis, processes[i])
    plt.show()
####################################################################################################    
def brownian_motion_log_returns(param):
    """
    This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
    about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
    :param param: the model parameters object
    :return: brownian motion log returns
    """
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.all_time)
####################################################################################################
def brownian_motion_levels(param):
    """
    Returns a price sequence whose returns evolve according to a brownian motion
    :param param: model parameters object
    :return: returns a price sequence which follows a brownian motion
    """
    return convert_to_prices(param, brownian_motion_log_returns(param))
####################################################################################################    
def geometric_brownian_motion_log_returns(param):
    """
    This method constructs a sequence of log returns which, when exponentiated, produce a random Geometric Brownian
    Motion (GBM). GBM is the stochastic process underlying the Black Scholes options pricing formula.
    :param param: model parameters object
    :return: returns the log returns of a geometric brownian motion process
    """
    assert isinstance(param, ModelParameters)
    wiener_process = numpy.array(brownian_motion_log_returns(param))
    sigma_pow_mu_delta = (param.gbm_mu - 0.5 * math.pow(param.all_sigma, 2.0)) * param.all_delta
    return wiener_process + sigma_pow_mu_delta
####################################################################################################
def geometric_brownian_motion_levels(param):
    """
    Returns a sequence of price levels for an asset which evolves according to a geometric brownian motion
    :param param: model parameters object
    :return: the price levels for the asset
    """
    return convert_to_prices(param, geometric_brownian_motion_log_returns(param))
####################################################################################################    
def jump_diffusion_process(param):
    """
    This method produces a sequence of Jump Sizes which represent a jump diffusion process. These jumps are combined
    with a geometric brownian motion (log returns) to produce the Merton model.
    :param param: the model parameters object
    :return: jump sizes for each point in time (mostly zeroes if jumps are infrequent)
    """
    assert isinstance(param, ModelParameters)
    s_n = time = 0
    small_lamda = -(1.0 / param.lamda)
    jump_sizes = []
    for k in range(0, param.all_time):
        jump_sizes.append(0.0)
    while s_n < param.all_time:
        s_n += small_lamda * math.log(random.uniform(0, 1))
        for j in range(0, param.all_time):
            if time * param.all_delta <= s_n * param.all_delta <= (j + 1) * param.all_delta:
                # print("was true")
                jump_sizes[j] += random.normalvariate(param.jumps_mu, param.jumps_sigma)
                break
        time += 1
    return jump_sizes
####################################################################################################
def geometric_brownian_motion_jump_diffusion_log_returns(param):
    """
    This method constructs combines a geometric brownian motion process (log returns) with a jump diffusion process
    (log returns) to produce a sequence of gbm jump returns.
    :param param: model parameters object
    :return: returns a GBM process with jumps in it
    """
    assert isinstance(param, ModelParameters)
    jump_diffusion = jump_diffusion_process(param)
    geometric_brownian_motion = geometric_brownian_motion_log_returns(param)
    return numpy.add(jump_diffusion, geometric_brownian_motion)
####################################################################################################
def geometric_brownian_motion_jump_diffusion_levels(param):
    """
    This method converts a sequence of gbm jmp returns into a price sequence which evolves according to a geometric
    brownian motion but can contain jumps at any point in time.
    :param param: model parameters object
    :return: the price levels
    """
    return convert_to_prices(param, geometric_brownian_motion_jump_diffusion_log_returns(param))
####################################################################################################
####################################################################################################
def hurst(ts):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0
####################################################################################################
def GetStatistics(path):
    df = pd.read_excel(path)
    y = df['Last'].values.ravel()
    X = np.arange(y.size)
    lm = LinearRegression(fit_intercept = True)
    lm.fit(X.reshape(-1,1), y)
    a = lm.coef_[0]
    b = lm.intercept_
    print a
    print b
    plt.figure()
    plt.plot(y - (b + a*X), color = 'black')
    print hurst(y - (b + a*X))
    adfuller_y = statsmodels.api.tsa.adfuller(y - (b + a*X))
    print 'Augmented Dickey Fuller test statistic =',adfuller_y[0]
    print 'Augmented Dickey Fuller p-value =',adfuller_y[1]
    print 'Augmented Dickey Fuller 1%, 5% and 10% test statistics =',adfuller_y[4]
    
    plt.figure()
    plotting.autocorrelation_plot(y)
    plt.title('autocorrelation of LAST values')
    
    rm = []
    for i in range(1, y.size):
        rm.append(np.mean(y[:i]))
        
    plt.figure()
    plt.plot(y)
    plt.title('LAST')
    plt.plot(np.array(rm))
    plt.plot(b + a*X)
    plt.title('detrended LAST')
    
    rsigma = []
    rstx = []
    for i in range(1, y.size):
        xt = y[i-1]
        sigmatsx = np.std(y[:i])
        rstx.append(sigmatsx)
        if y[i] != xt:
            rsigma.append((y[i] - xt)/sigmatsx)
        else:
            rsigma.append(0)
    
    plt.figure()
    plt.plot(np.array(rsigma), color = 'orange')    
    plt.title('how many std dev the price has moved')
    plt.figure()
    plt.plot(np.array(rstx))
    plt.title('rolling std dev')
    plt.figure()
    plotting.autocorrelation_plot(np.array(rstx), color = 'green')
    plt.title('rolling std dev autocorrelation')    
    plt.figure()
    plotting.autocorrelation_plot(np.array(rsigma[1:]), color = 'palevioletred')
    plt.title('autocorrelation "skewing std dev"')
    
    ratio = []
    for i in range(1, len(rsigma)):
        ratio.append(float(np.where(np.array(rsigma[:i]) <= 0)[0].size)/float(np.array(rsigma[:i]).size))
    plt.figure()
    plt.plot(np.array(ratio))
    plt.title('ratio of std dev downward or upward movements')
    
    print 'percentage downward movements: {}'.format(float(np.where(np.array(rsigma[1:]) <= 0)[0].size)/float(np.array(rsigma[1:]).size))
    print 'skewing std dev mean: {}'.format(np.mean(rsigma[1:]))
    print 'skewing std dev max: {}'.format(np.max(rsigma[1:]))
    print 'skewing std dev min: {}'.format(np.min(rsigma[1:]))
####################################################################################################

ger = pd.read_excel('C:/Users/utente/Documents/Trading/Chiusura GER_15_16.xlsx')
ger.columns = [['Data', 'CAL', 'YEAR']]

ger2015 = ger.ix[ger['YEAR'] == 2015]
ger2016 = ger.ix[ger['YEAR'] == 2016]


levy_ger15 = scipy.stats.levy.fit(ger2015['CAL'])

plt.figure()
ger2015['CAL'].plot()
plt.figure()
ger2015['CAL'].plot(kind = 'hist', bins = 20)
plt.figure()
ger2016['CAL'].plot(color = 'violet')
plt.figure()
ger2016['CAL'].plot(kind = 'hist', bins = 20, color = 'violet')


sim15 = scipy.stats.levy.rvs(levy_ger15[0], levy_ger15[1], size = 5*ger2015.shape[0])

sim15[np.where(sim15 > ger2015['CAL'].max())[0]] = ger2015['CAL'].max()

plt.figure()
plt.hist(sim15, bins = 20, color = 'teal')
plt.figure()
plt.plot(sim15, color = 'teal')

print hurst(ger2015['CAL'].values.ravel())
print hurst(ger2016['CAL'].values.ravel())

adfuller2015 = statsmodels.api.tsa.adfuller(ger2015['CAL'].values.ravel())
adfuller2016 = statsmodels.api.tsa.adfuller(ger2016['CAL'].values.ravel())

print 'Augmented Dickey Fuller test statistic =',adfuller2015[0]
print 'Augmented Dickey Fuller p-value =',adfuller2015[1]
print 'Augmented Dickey Fuller 1%, 5% and 10% test statistics =',adfuller2015[4]
print 'Augmented Dickey Fuller test statistic =',adfuller2016[0]
print 'Augmented Dickey Fuller p-value =',adfuller2016[1]
print 'Augmented Dickey Fuller 1%, 5% and 10% test statistics =',adfuller2016[4]

y = ger2015['CAL'].values.ravel()
X = np.arange(y.size)
lm = LinearRegression(fit_intercept = True)
lm.fit(X.reshape(-1,1), y)
a = lm.coef_[0]
b = lm.intercept_
print a
print b
plt.figure()
plt.plot(y - (b + a*X), color = 'black')
print hurst(y - (b + a*X))
adfuller_y = statsmodels.api.tsa.adfuller(y - (b + a*X))
print 'Augmented Dickey Fuller test statistic =',adfuller_y[0]
print 'Augmented Dickey Fuller p-value =',adfuller_y[1]
print 'Augmented Dickey Fuller 1%, 5% and 10% test statistics =',adfuller_y[4]


y = ger2016['CAL'].values.ravel()
X = np.arange(y.size)
lm = LinearRegression(fit_intercept = True)
lm.fit(X.reshape(-1,1), y)
a = lm.coef_[0]
b = lm.intercept_
print a
print b
plt.figure()
plt.plot(y - (b + a*X), color = 'black')
print hurst(y - (b + a*X))
adfuller_y = statsmodels.api.tsa.adfuller(y - (b + a*X))
print 'Augmented Dickey Fuller test statistic =',adfuller_y[0]
print 'Augmented Dickey Fuller p-value =',adfuller_y[1]
print 'Augmented Dickey Fuller 1%, 5% and 10% test statistics =',adfuller_y[4]

plt.figure()
plotting.autocorrelation_plot(y)

rm = []
for i in range(1, y.size):
    rm.append(np.mean(y[:i]))
    
plt.figure()
plt.plot(y)
plt.plot(np.array(rm))
plt.plot(b + a*X)

rsigma = []
rstx = []
for i in range(1, y.size):
    xt = y[i-1]
    sigmatsx = np.std(y[:i])
    rstx.append(sigmatsx)
    if y[i] != xt:
        rsigma.append((y[i] - xt)/sigmatsx)
    else:
        rsigma.append(0)

plt.figure()
plt.plot(np.array(rsigma), color = 'orange')    
plt.figure()
plt.plot(np.array(rstx))
plt.figure()
plotting.autocorrelation_plot(y, color = 'coral')
plt.figure()
plotting.autocorrelation_plot(np.array(rstx), color = 'green')
plt.figure()
plotting.autocorrelation_plot(np.array(rsigma[1:]), color = 'palevioletred')

ratio = []
for i in range(1, len(rsigma)):
    ratio.append(float(np.where(np.array(rsigma[:i]) <= 0)[0].size)/float(np.array(rsigma[:i]).size))
plt.figure()
plt.plot(np.array(ratio))
#### Try fitting an Ornstein–Uhlenbeck process

plt.figure()
plotting.autocorrelation_plot(ger2015['CAL'].values.ravel())
plt.figure()
plotting.lag_plot(ger2015['CAL'])

X = ger2015['CAL'].values.ravel()[:-1]
y = ger2015['CAL'].values.ravel()[1:]

lm = LinearRegression(fit_intercept = True)

lm.fit(X.reshape(-1,1), y)

a = lm.coef_[0]
b = lm.intercept_

Sxy = np.sum(X*y)
Sx = np.sum(X)
Sy = np.sum(y)
Sxx = np.sum(X**2)
Syy = np.sum(y**2)
n = X.size

a = (n*Sxy - Sx*Sy)/(n*Sxx - Sx**2)
b = (Sy - a*Sx)/(n)
sdepsilon = np.sqrt((n*Syy - Sy**2 - a*(n*Sxy - Sx*Sy))/(n*(n-2)))

residuals = y - lm.predict(X.reshape(-1,1))

lam = -np.log(a)
mu = b/(1 - a)
sigma = sdepsilon*np.sqrt(2*lam/(1 - a**2))

S0 = np.mean(X)
S = [S0]
for i in range(1,229):
    rnv = scipy.stats.norm.rvs()
    s = lam * (mu - S[i-1]) + sigma * rnv + np.mean(X)
    S.append(s)
    
plt.figure()
plt.plot(np.array(S), color = 'lawngreen')


####################################################################################################

GetStatistics('H:/Energy Management/13. TRADING/GER_1718.xlsx')

path = 'H:/Energy Management/13. TRADING/GER_1718.xlsx'
df = pd.read_excel(path)
X = df['Last'].values.ravel()

plt.figure()
plt.plot(np.diff(X))
plt.axhline(y = scipy.stats.mstats.mquantiles(np.diff(X), prob = 0.025), color = 'black')
plt.axhline(y = scipy.stats.mstats.mquantiles(np.diff(X), prob = 0.975), color = 'black')

Xd = np.diff(X)
Xd = Xd[Xd > np.min(Xd)]
plt.figure()
plt.hist(Xd, bins = 20, color = 'yellow')
plt.figure()
plt.hist(scipy.stats.cauchy.rvs(size = Xd.size), bins = 20, color = 'midnightblue')
plt.figure()
plt.hist(scipy.stats.norm.rvs(size = Xd.size), bins = 20, color = 'salmon')

print scipy.stats.mstats.normaltest(Xd)
print scipy.stats.shapiro(Xd)
print scipy.stats.kstest(Xd,scipy.stats.tstd)
print scipy.stats.kstest(Xd,scipy.stats.cauchy)

res = X - (0.0190793255721 * np.arange(X.size) + 23.6157603774)
plt.figure()
plt.plot(res, color = 'crimson')
plt.figure()
plotting.autocorrelation_plot(res)

eta = 0.0036
lam = 10
sigma = np.std(res)

S0 = np.mean(X)
S = [S0]
for i in range(X.size):
    rnv = scipy.stats.norm.rvs(scale=sigma**2)
    Xbar = lambda i: 0.0190793255721 * i + 23.6157603774
    s = eta * (Xbar(i) - S[i-1]) + rnv + scipy.stats.bernoulli.rvs(p = 1/X.size) + S0
    S.append(s)
    
plt.figure()
plt.plot(np.array(S), color = 'plum')


mp = ModelParameters(all_s0=X[-1],
                     all_r0=0.5,
                     all_time=800,
                     all_delta=0.00396825396,
                     all_sigma=0.125,
                     gbm_mu=0.058,
                     jumps_lamda=float(1.0/450.0),
                     jumps_sigma=0.0001,
                     jumps_mu=-0.10,
                     cir_a=3.0,
                     cir_mu=0.5,
                     cir_rho=0.5,
                     ou_a=0.36,
                     ou_mu=0.5,
                     heston_a=0.25,
                     heston_mu=0.35,
                     heston_vol0=0.06125)

paths = 100

jump_diffusion_examples = []
for i in range(paths):
    jump_diffusion_examples.append(geometric_brownian_motion_jump_diffusion_levels(mp))
plot_stochastic_processes(jump_diffusion_examples, "Jump Diffusion Geometric Brownian Motion (Merton)")

final = []
for i in range(100):
    final.append(jump_diffusion_examples[i][-1])
final = np.array(final)

print np.mean(final)
print np.std(final)
print scipy.stats.skew(final)
print scipy.stats.kurtosis(final)

plt.figure()
plt.hist(final, bins = 20, color = 'deepskyblue')