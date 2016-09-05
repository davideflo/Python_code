# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:11:16 2016

@author: d_floriello

Script for pattern analysis in PUN time series
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy as sp
#from scipy.interpolate import interp1d
from sklearn import linear_model
#import mpmath as mp
from matplotlib.legend_handler import HandlerLine2D
import statsmodels
from pandas.tools.plotting import lag_plot


data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")
data2 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2011.xlsx")
data3 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2012.xlsx")
data4 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2013.xlsx")
data5 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2014.xlsx")
data6 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2015.xlsx")

data7 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_07.xlsx", sheetname = 1)

pun = data["PUN"]
pun = data["CSUD"]
#dates = temp.dates(data[data.columns[0]])
#prova = pd.to_datetime(dates)
df = pd.DataFrame(pun)
rng = pd.date_range('01/01/2010', periods=8760, freq='H')
df = df.set_index(rng)

dec = sm.tsa.seasonal_decompose(df.values,freq=24)
dec.plot()

s = dec.seasonal

min_season = np.array(s[0:24])
plt.plot(min_season)

rmse = []
for i in range(24, 8760, 24):
    ns = s[i:i+24]
    rmse.append(np.sqrt(np.mean((min_season - ns)**2)))
### seasonal component is identical during every day of the year

random_hour = np.random.randint(0,8760,1)    
    
initial = np.array(pun[0:24])
random_pun = np.array(pun[random_hour[0]:random_hour[0]+24])

plt.plot(initial)
plt.plot(random_pun)

###############################################
def deseasonalise(x, min_s, freq):
    x_ds = []
    for i in range(0, x.size, freq):
        x_ds.append(x[i:i+24] - min_s)
    x_ds = np.array(x_ds).flatten()
    return x_ds
##############################################

des = deseasonalise(pun, min_season, 24)     
    
dec_des = sm.tsa.seasonal_decompose(des,freq=24)
dec_des.plot()
dec.plot()

sds = dec_des.seasonal

min_season_des = np.array(sds[0:24])
plt.plot(min_season_des)

x = np.array(range(24))

new_length = 100
new_x = np.linspace(0, 23, new_length)
fhat = sp.interpolate.interp1d(x, min_season, kind='cubic')

plt.plot(new_x,fhat(new_x))
plt.plot(min_season)

tr = dec.trend
plt.plot(tr)

plt.subplot(2,1,1)
plt.plot(tr)
plt.subplot(2,1,2)
plt.plot(pun)

####### empirical risk analysis ######
(np.max(pun) - np.mean(pun))/np.std(pun)
 
index_max = pun.tolist().index(np.max(pun)) 
rng[index_max]
 
## how many times the values are > mean+x*sigma?
#################################################
def freq_greater_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int((x - np.mean(ts))/np.std(ts) > sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
################################################# 
def freq_smaller_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int((x - np.mean(ts))/np.std(ts) < -sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
#################################################
def abs_freq_greater_than(ts, sig, flag):
    greater = []
    for x in ts:
        greater.append(int(abs(x - np.mean(ts))/np.std(ts) > sig))
    greater = np.array(greater)
    if flag:
        return greater
    else:
        return np.sum(greater)/greater.size
#################################################
for x in range(1,7,1):
   print(x)
   print( '%.6f' % freq_greater_than(pun, x, False))

sigma_4 = freq_greater_than(pun, 4, True)
out_hours = np.where(sigma_4 > 0)[0] 

count = 0
for i in range(out_hours.size - 1):
    if out_hours[i+1] - out_hours[i] == 1:
        count += 1
    else:
        pass
################################################
def rolling_mean_at(ts, time_interval):
    tsm = []
    for i in range(0, ts.size, time_interval):
        tsm.append(np.mean(ts[i:i+time_interval]))
    return np.array(tsm)
################################################
m24 = rolling_mean_at(pun, 24)    
m2 = rolling_mean_at(pun, 2)

plt.subplot(2,1,1)
plt.plot(m24)
plt.subplot(2,1,2)
plt.plot(tr)
################################################
### NOT RUN
def adaptive_trend(ts, m, n, I, O, coefs,epsilon=1e-06):
    y0 = np.array(ts[m:n+1])
    x0 = np.linspace(m,n,y0.size)
    model0 = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 20)
    model0.fit(x0.reshape([x0.size,1]), y0)
    coefs.append(model0.estimator_.coef_)
    for i in range(n+1, ts.size, 1):
        xnew = ts[i]
        ynew = model0.predict(i)
        error = abs(ynew - xnew)
        if error <= epsilon:
            I.append(i)
        else:
            O.append(i)
            adaptive_trend(ts, i, i+1, I, O, coefs, epsilon)
    return 0
###############################################    
I = []
O = []
coefs = []   
epsilon = 20
adaptive_trend(pun, 0, 2000, I, O, coefs, epsilon)    
### END: NOT RUN

ret = []
for i in range(0, pun.size - 1, 1):
    ret.append((pun[i+1] - pun[i])/pun[i])
    
plt.subplot(2,1,1)
plt.plot(pun)
plt.subplot(2,1,2)
plt.plot(ret)

### does there exist a "signal" that something is going to happen? e.g.: is there anything suggesting the trend is changing?

### some analyses on the whole dataset ###
varn = "PUN"

pun = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
                               np.array(data3[varn]),
                         np.array(data4[varn]), np.array(data5[varn]), 
                            np.array(data6[varn])])

for x in range(1,10,1):
   print(x)
   print( '%.6f' % freq_greater_than(pun, x, False))

sigma = freq_greater_than(pun, 4, True)
out_hours = np.where(sigma > 0)[0] 

count = 0
for i in range(out_hours.size - 1):
    if out_hours[i+1] - out_hours[i] == 1:
        count += 1
    else:
        pass

### percentage of two consecutive peaks for "normalised distances" (Mahalanobis?) greater than 4
count/np.sum(sigma)

### percentage of two consecutive peaks for all levels of norm. distances:
def glob_perc(ts):
    res = []
    for x in range(1, 10, 1):
        sigma = freq_greater_than(ts, x, True)
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            count = 0
            for i in range(out.size - 1):
                if out[i+1] - out[i] == 1:
                    count += 1
                else:
                    pass
            res.append(float(count/np.sum(sigma)))
            print('at distance {}'.format(x))    
            print('%.6f' % float(count/np.sum(sigma)))
    return np.array(res)
###########################################################        
glob_perc(pun)
###########################################################
def glob_perc_neg(ts):
    res = []
    for x in range(1, 10, 1):
        sigma = freq_smaller_than(ts, x, True)
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            count = 0
            for i in range(out.size - 1):
                if out[i+1] - out[i] == 1:
                    count += 1
                else:
                    pass
            res.append(float(count/np.sum(sigma)))
            print('at distance {}'.format(x))    
            print('%.6f' % float(count/np.sum(sigma)))
    return np.array(res) 
###########################################################
def cumulative_glob_perc(ts, period, step):
    perc = []
    year = lambda y: np.ceil(y/step)
    for j in range(0, ts.size, step):
        start = np.choose(j-period >0, [0, j-period])
        ts2 = ts[start:j]
        for x in range(1, 10, 1):
            sigma = freq_greater_than(ts2, x, True)
            out = np.where(sigma > 0)[0]
            if np.sum(sigma) > 0:
                count = 0
                for i in range(out.size - 1):
                    if out[i+1] - out[i] == 1:
                        count += 1
                    else:
                        pass
                    #print(mp.mpf(count))
                    #print(mp.mpf(np.sum(sigma)))
                print('after {} year, at distance {}'.format(year(j), x)) 
                print('%.6f' % float(count/np.sum(sigma)))
                perc.append(float(count/np.sum(sigma)))
    return np.array(perc)
                    #print('%.6f' % float(mp.mpf(count)/mp.mpf(np.sum(sigma))))
#############################################################    
per = cumulative_glob_perc(pun, 8760, 8760)

### markers for matplotlib: http://stackoverflow.com/questions/8409095/matplotlib-set-markers-for-individual-points-on-a-line
plt.plot(per,linestyle='--', marker='o')

##### manual: 
res = glob_perc(data[varn])
res2 = glob_perc(data2[varn])
res3 = glob_perc(data3[varn])
res4 = glob_perc(data4[varn])
res5 = glob_perc(data5[varn])
res6 = glob_perc(data6[varn])
res7 = glob_perc(data7[varn])

plt.figure()
line1, = plt.plot(np.array(range(1,res.size+1,1)), res, linewidth = 2, marker='o', label='Line 2010')
line2, = plt.plot(np.array(range(1,res2.size+1,1)), res2, linewidth = 2, marker='o', label='Line 2011')
line3, = plt.plot(np.array(range(1,res3.size+1,1)), res3, linewidth = 2, marker='o', label='Line 2012')
line4, = plt.plot(np.array(range(1,res4.size+1,1)), res4, linewidth = 2, marker='o', label='Line 2013')
line5, = plt.plot(np.array(range(1,res5.size+1,1)), res5, linewidth = 2, marker='o', label='Line 2014')
line6, = plt.plot(np.array(range(1,res6.size+1,1)), res6, linewidth = 2, marker='o', label='Line 2015')
line7, = plt.plot(np.array(range(1,res7.size+1,1)), res7, linewidth = 2, marker='o', label='Line 2016')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line2: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line4: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line5: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line6: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line7: HandlerLine2D(numpoints=2)})

plt.xlabel('Normalised distance')
plt.ylabel('Frequency (Probability)')
plt.title(r'Percentage of 2 consecutives peaks (in absolute value) at a given distance')

#### negative peaks:
res = glob_perc_neg(data[varn])
res2 = glob_perc_neg(data2[varn])
res3 = glob_perc_neg(data3[varn])
res4 = glob_perc_neg(data4[varn])
res5 = glob_perc_neg(data5[varn])
res6 = glob_perc_neg(data6[varn])
res7 = glob_perc_neg(data7[varn])

plt.figure()
line1, = plt.plot(np.array(range(1,res.size+1,1)), res, linewidth = 2, marker='o', label='Line 2010')
line2, = plt.plot(np.array(range(1,res2.size+1,1)), res2, linewidth = 2, marker='o', label='Line 2011')
line3, = plt.plot(np.array(range(1,res3.size+1,1)), res3, linewidth = 2, marker='o', label='Line 2012')
line4, = plt.plot(np.array(range(1,res4.size+1,1)), res4, linewidth = 2, marker='o', label='Line 2013')
line5, = plt.plot(np.array(range(1,res5.size+1,1)), res5, linewidth = 2, marker='o', label='Line 2014')
line6, = plt.plot(np.array(range(1,res6.size+1,1)), res6, linewidth = 2, marker='o', label='Line 2015')
line7, = plt.plot(np.array(range(1,res7.size+1,1)), res7, linewidth = 2, marker='o', label='Line 2016')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line2: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line4: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line5: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line6: HandlerLine2D(numpoints=2)})
plt.legend(handler_map={line7: HandlerLine2D(numpoints=2)})

plt.xlabel('Normalised distance')
plt.ylabel('Frequency (Probability)')
plt.title(r'Percentage of 2 consecutives negative peaks at a given distance')

#### what is the average distance between peaks? 
def compute_average_distance_between_peaks(ts, flag_s):
    dist = []
    for x in range(1, 10, 1):
        #sigma = freq_greater_than(ts, x, True)
        sigma = abs_freq_greater_than(ts, x, True)        
        out = np.where(sigma > 0)[0]
        if np.sum(sigma) > 0:
            dist_sigma = []
            for i in range(out.size-1):
                dist_sigma.append(out[i+1]-out[i])
            dist.append(np.nanmean(dist_sigma))
    if flag_s:
        return dist_sigma
    else:
        return np.array(dist)
###############################################
dp = compute_average_distance_between_peaks(pun, False)            

av = compute_average_distance_between_peaks(data[varn], False)            
av2 = compute_average_distance_between_peaks(data2[varn], False)            
av3 = compute_average_distance_between_peaks(data3[varn], False)            
av4 = compute_average_distance_between_peaks(data4[varn], False)            
av5 = compute_average_distance_between_peaks(data5[varn], False)            
av6 = compute_average_distance_between_peaks(data6[varn], False)            
av7 = compute_average_distance_between_peaks(data7[varn], False)            

peaks_dist = {'2010': av,'2011': av2,'2012': av3,'2013': av4,'2014': av5,
              '2015': av6,'2016': av7}

dfp = pd.DataFrame.from_dict(peaks_dist, orient='index')

dfp = dfp.transpose().set_index([[1,2,3,4,5,6,7,8]]).fillna(0)

cols = dfp.columns.tolist()
#ord_cols = [cols[2] , cols[4] , cols[1] , cols[0] , cols[3], cols[5], cols[6]]
ord_cols = [cols[4] , cols[1] , cols[3] , cols[5] , cols[0], cols[6], cols[2]]
dfp = dfp[ord_cols]

dfp.mean(axis=0)
dfp.mean(axis=0)/24
dfp.mean(axis=1)
dfp.mean(axis=1)/24

dfp.plot(marker='o', title='distance in hours between consecutive peaks at a given norm. distance')
dfp.transpose().plot(marker='o', title = 'distance in hours between consecutive peaks at given year')
(dfp/24).plot(marker='o', title='distance in days between consecutive peaks at a given norm. distance')
(dfp/24).transpose().plot(marker='o', title = 'distance in days between consecutive peaks at given year')

tot = np.nan_to_num(np.concatenate([av,av2,av3,av4,av5,av6,av7]))/24

plt.figure()
plt.plot(tot, marker='o')

plt.figure()
plt.plot(statsmodels.tsa.stattools.periodogram(tot))

################################################
def fourierExtrapolation(x, n_predict):
    x = np.array(x)
    n = x.size
    n_harm = 100                     # number of harmonics in model
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
################################################

rec_tot = fourierExtrapolation(tot, 63)

plt.figure()
plt.plot(rec_tot)

###################################################################
### does the mean exhibit noticeable monthwise changes?
varn = "PUN"

pun = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
                               np.array(data3[varn]),
                         np.array(data4[varn]), np.array(data5[varn]), 
                            np.array(data6[varn]),np.array(data7[varn])])

df = pd.DataFrame(pun)
rng = pd.date_range('01/01/2010', periods=pun.size, freq='H')
df = df.set_index(rng)

letters = 'abcdefghilmnopq'
lets = []
for i in range(rng.size):
    lets.append(letters[rng[i].month])

letters_dict = {'Letters': lets, 'pun': pun}
ldf = pd.DataFrame(letters_dict).set_index(rng)


dfbm = pd.groupby(df,by=[df.index.month,df.index.year])
dfbm = pd.groupby(ldf,by='Letters')

dfbm2 = pd.DataFrame(pun).set_index(lets)
dfbm2.plot.box()

jan = ldf.ix[ldf['Letters'] == 'b']

monthwise = {}

for i in range(1,13,1):
    letter = letters[i]
    mese = ldf.ix[ldf['Letters'] == letter]
    yearly_mean = []
    for j in range(2010,2016,1):
        year = mese.ix[mese.index.year == j]
        yearly_mean.append(np.mean(year['pun']))
    monthwise[letter] = yearly_mean

monthwise = {}

for i in range(1,13,1):
    letter = letters[i]
    mese = ldf.ix[ldf['Letters'] == letter]
    yearly_mean = []
    for j in range(2010,2017,1):
        year = mese.ix[mese.index.year == j]
        yearly_mean.append(np.std(year['pun']))
    monthwise[letter] = yearly_mean


bymonth2 = pd.DataFrame(monthwise).transpose()
bymonth2.columns = [['2010', '2011', '2012', '2013', '2014', '2015','2016']]
bymonth2 = bymonth2.set_index([['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov' ,'dec']])

bymonth2.transpose().plot(title='average prices behaviour along years')

### for bootstrap: do I sample only from the past year, same month?


for i in range(12):
    for j in range(5):
        print("difference between {} and {} for month {}".format(bymonth2.columns[j+1], bymonth2.columns[j], 
              bymonth2.index[i]))
        #print(bymonth2.apply(lambda j: bymonth2[bymonth2.columns[j+1]].ix[i] - bymonth2[bymonth2.columns[j]].ix[i],axis=1))
        print(bymonth2[bymonth2.columns[j+1]].ix[i] - bymonth2[bymonth2.columns[j]].ix[i])


from pandas.tools.plotting import autocorrelation_plot

for i in range(12):
    plt.figure()
    autocorrelation_plot(bymonth2.ix[i])

### hour-by-hour
letters = 'abcdefghijklmnopqrstuvwxyz'
lets = []
for i in range(rng.size):
    lets.append(letters[rng[i].hour])

letters_dict = {'Letters': lets, 'csud': pun}
hdf = pd.DataFrame(letters_dict).set_index(rng)

hourwise= {}

for i in range(24):
    letter = letters[i]
    hour = hdf.ix[hdf['Letters'] == letter]
    hourly_mean = []
    for j in range(2010,2016,1):
        year = hour.ix[hour.index.year == j]
        hourly_mean.append(np.mean(year['csud']))
    hourwise[letter] = hourly_mean

HDF = pd.DataFrame(hourwise).transpose()
HDF.columns =  [['2010', '2011', '2012', '2013', '2014', '2015']]
HDF = HDF.reset_index(drop=True)

##########################################################################
##### analysis of prices on daylight saving dates
varn = "PUN"

pun = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
                               np.array(data3[varn]),
                         np.array(data4[varn]), np.array(data5[varn]), 
                            np.array(data6[varn]), np.array(data7[varn])])

ora = data.columns[1]

hours = np.concatenate([np.array(data[ora]), np.array(data2[ora]), 
                               np.array(data3[ora]),
                         np.array(data4[ora]), np.array(data5[ora]), 
                            np.array(data6[ora]), np.array(data7[ora])])

ph = {"hour": hours, "pun": pun}

phdf = pd.DataFrame(ph)

idx = []
diff = []
for i in range(phdf.shape[0]-1):
    if phdf['hour'].ix[i] == 23 and phdf['hour'].ix[i+1] == 24:
        idx.append(i)
        #idx.append(i+1)
        diff.append(phdf['pun'].ix[i+1] - phdf['pun'].ix[i])        
#########################################################################
#########################################################################
### in the fixed models built with h2o and R, the models for 9 PM have all a low R2 ###
#########################################################################
def Extract_Hour(hour):
    data = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2010.xlsx")
    data2 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2011.xlsx")
    data3 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2012.xlsx")
    data4 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2013.xlsx")
    data5 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2014.xlsx")
    data6 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2015.xlsx")
    
    data7 = pd.read_excel("C:/Users/d_floriello/Documents/PUN/Anno 2016_07.xlsx", sheetname = 1)

    nine1 = [data["PUN"].ix[i] for i in range(data.shape[0]) if data[data.columns[1]].ix[i] == hour]         
    nine2 = [data2["PUN"].ix[i] for i in range(data2.shape[0]) if data2[data2.columns[1]].ix[i] == hour]         
    nine3 = [data3["PUN"].ix[i] for i in range(data3.shape[0]) if data3[data3.columns[1]].ix[i] == hour]         
    nine4 = [data4["PUN"].ix[i] for i in range(data4.shape[0]) if data4[data4.columns[1]].ix[i] == hour]         
    nine5 = [data5["PUN"].ix[i] for i in range(data5.shape[0]) if data5[data5.columns[1]].ix[i] == hour]         
    nine6 = [data6["PUN"].ix[i] for i in range(data6.shape[0]) if data6[data6.columns[1]].ix[i] == hour]         
    nine7 = [data7["PUN"].ix[i] for i in range(data7.shape[0]) if data7[data7.columns[1]].ix[i] == hour]         
            
    print(max(nine1))
    print(max(nine2))
    print(max(nine3))
    print(max(nine4))
    print(max(nine5))
    print(max(nine6))
    print(max(nine7))
        
    nine = np.concatenate([np.array(nine1), np.array(nine2), np.array(nine3), np.array(nine4),
                           np.array(nine5), np.array(nine6), np.array(nine7)])        
        
    Nine = pd.DataFrame(nine)

    return Nine
##########################################################################

#for i in range(19,22,1):
#    D = Extract_Hour(i)
#    D.plot()
#    plt.figure()
#    lag_plot(D)


D = Extract_Hour(21)
D.plot()
lag_plot(D)

x21 = data3["PUN"].ix[data3[data3.columns[1]] == 21]
for x,i in enumerate(x21):
    print(x)
    print(i)
    if(i == max(x21)):
        break 
###################################################################################
###################################################################################
### hourwise patterns ###

names = ['data','data2','data3','data4','data5','data6','data7']        
d = {}
d2 = {}

for n in names:
    D = locals()[n]
    hm = []
    hv = []
    for h in range(1,25,1):
        hm.append(D["PUN"].ix[D[D.columns[1]] == h].mean())
        hv.append(D["PUN"].ix[D[D.columns[1]] == h].std())
    d[n] = hm
    d2[n] = hv

HD = pd.DataFrame.from_dict(d)
HD.columns = ['2010','2011','2012','2013','2014','2015','2016']
HV = pd.DataFrame.from_dict(d2)
HV.columns = ['2010','2011','2012','2013','2014','2015','2016']

diff_mean_2016 = HD[HD.columns[:6]].mean(axis=1) - HD['2016']

#####################################################################################
#####################################################################################
#### probability of moving upwards or downwards ####

from sklearn.neighbors.kde import KernelDensity

def hkde(bandwidth, hour, ln):
    wh = []
    if not isinstance(ln, str):
        for n in ln:
            D = globals()[n]
            wh.append(D['PUN'].ix[D[D.columns[1]] == hour])
        wh2 = [val for sublist in wh for val in sublist]    
        wh = np.array(wh2)
        kdew = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(wh.reshape(-1,1))
        return kdew
    else:
        D = globals()[ln]
        wh.append(D['PUN'].ix[D[D.columns[1]] == hour])
        wh2 = [val for sublist in wh for val in sublist]    
        wh = np.array(wh2)
        kdew = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(wh.reshape(-1,1))
        return kdew
######################################################################################
for h in range(1,50,1):
    kd = hkde(h,1,names[:6])
    print(kd.score(np.linspace(start=0,stop=150,num=1000).reshape(-1,1)))

distr_15 = hkde(4,1,names[:6])                
xplot = np.linspace(start=0,stop=200,num=1000)
yplot = np.exp(distr_15.score_samples(xplot.reshape(-1,1)))        
plt.figure() 
plt.plot(xplot,yplot)       

from scipy.stats import mode
print(mode(pun))

        
        
distr_16 = hkde(4,1,'data7')        
        
yplot2 = np.exp(distr_16.score_samples(xplot.reshape(-1,1)))        
        
plt.figure() 
plt.plot(xplot,yplot2)       

fig, ax = plt.subplots()        
ax.plot(xplot,yplot2)
ax.plot(xplot,yplot)        
        
import scipy.integrate as integrate
######################################################################################
def compute_probability(low, up, distr):
#    x = np.linspace(start=low,stop=up,num=1000)
#    logy = distr.score_samples(x.reshape(-1,1))
    def distribution(x,distr):
        x = np.array(x)
        return np.exp(distr.score_samples(x.reshape(-1,1)))
#     quad wants a single value as first argument???
    J = integrate.quad(distribution,low,up, args = (distr,))  
#    I = integrate.quad(lambda x: np.exp(logy),low,up, args = x)
    return J
######################################################################################
def Expected_Loss_inf(v, distr):
    
    def f(x,v,distr):
        x = np.array(x)
        return ((x - v) ** 2) * np.exp(distr.score_samples(x.reshape(-1,1)))
        
    J = integrate.quad(f, 0, v, args = (v,distr))
    return J
######################################################################################
def Expected_Loss_sup(v, distr):
    
    def f(x,v,distr):
        x = np.array(x)
        return ((x - v) ** 2) * np.exp(distr.score_samples(x.reshape(-1,1)))
        
    J = integrate.quad(f, v, np.inf, args = (v,distr))
    return J
######################################################################################
compute_probability(50.17, 200, distr_15)
compute_probability(50.17, 200, distr_16)

print(compute_probability(50.17, np.inf, distr_15))
print(compute_probability(50.17, np.inf, distr_16))


print(Expected_Loss_inf(40.00, distr_15))
print(Expected_Loss_sup(40.00, distr_15))

print(Expected_Loss_inf(40.00, distr_16))
print(Expected_Loss_sup(40.00, distr_16))

#########################################################################################
############ trend and seasonality analysis in 2016 #####################################

pun16 = pd.DataFrame(data7['PUN'])
decp = sm.tsa.seasonal_decompose(pun16.values,freq=24)

decp.plot()

seas = decp.seasonal
plt.figure()
plt.plot(np.arange(1,seas[0:23].size+1,1),seas[0:23])

diffs = []
for i in range(23, seas.size-23,23):
    #print(i)
    diffs.append(np.sum(seas[i:i+23] - seas[i-23:i]))
    print(np.sum(seas[i:i+23] - seas[i-23:i]))

ds = pd.DataFrame(seas)
seaslist = [x for x in seas]
plt.figure()
plt.plot(np.linspace(0,1,num=len(seaslist[0:23])),seaslist[0:23])

jan = pun16.ix[0:24*31]
jul = pun16.ix[4366:].reset_index(drop=True)

jan_dec = sm.tsa.seasonal_decompose(jan.values,freq=24)
jul_dec = sm.tsa.seasonal_decompose(jul.values,freq=24)

jan_dec.plot()
jul_dec.plot()

jan.mean()
jul.mean()

plt.figure()
plt.plot(jul_dec.seasonal[0:23] - jan_dec.seasonal[0:23])

plt.figure()
plt.plot(jul_dec.seasonal[0:23])
plt.figure()
plt.plot(jan_dec.seasonal[0:23])
#####  #####  ##### ##### 
jan = data7.ix[0:743]
jul = data7.ix[4367:].reset_index(drop=True)

from collections import OrderedDict

dhjan = OrderedDict()
dhjul = OrderedDict()

for h in range(1,25,1):
    print(jan["PUN"].ix[jan[jan.columns[1]] == h].size)
    print(jul["PUN"].ix[jul[jul.columns[1]] == h].size)
    dhjan['ora-'+str(h)] = jan["PUN"].ix[jan[jan.columns[1]] == h].reset_index(drop=True)
    dhjul['ora-'+str(h)] = jul["PUN"].ix[jul[jul.columns[1]] == h].reset_index(drop=True)

hjan = pd.DataFrame.from_dict(dhjan, orient='columns')
hjul =pd.DataFrame.from_dict(dhjul)

plt.figure()
plt.plot(np.array(hjul.mean(axis=0)))

plt.figure()
plt.plot(np.array(hjul.mean(axis=0)) - np.mean(np.array(hjul.mean(axis=0))))

hjul.std(axis=0)

# # # # # # # # # # # # # # # # #

plt.figure()
plt.plot(np.array(hjan.mean(axis=0)))

plt.figure()
plt.plot(np.array(hjan.mean(axis=0)) - np.mean(np.array(hjan.mean(axis=0))))

hjan.std(axis=0)

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0,0].plot(np.array(hjul.mean(axis=0)))
ax[1,0].plot(np.array(hjan.mean(axis=0)))
ax[0,1].plot(seas[0:24])
ax[1,1].plot( 1/2*(np.array(hjul.mean(axis=0)) + np.array(hjan.mean(axis=0))) )

#############################################################################################

rng = pd.date_range('01/01/2016', periods=5111, freq='H')
pun16 = pun16.set_index(rng)

for i in range(1,8,1):
    mese = pun16.ix[pun16.index.month == i]
    decm = sm.tsa.seasonal_decompose(mese.values,freq=24)
    fig = plt.figure()
    sea = decm.seasonal
    plt.plot(sea[0:24])
    fig.suptitle('{} month'.format(i))



may = pun16.ix[pun16.index.month == 5]
jul = pun16.ix[pun16.index.month == 7]

may.mean()
jul.mean()

#### ####### ############### ########## ############# ##################

varn = "PUN"

pun = np.concatenate([np.array(data[varn]), np.array(data2[varn]), 
                               np.array(data3[varn]),
                         np.array(data4[varn]), np.array(data5[varn]), 
                            np.array(data6[varn])])
                            
                            
rng2 = pd.date_range('01/01/2010', periods=pun.size, freq='H')
pun = pun.set_index(rng2)

##### http://pselab.chem.polimi.it/pubblicazione/a-methodology-to-forecast-the-price-of-electric-energy/#
######  ###### ##### ##### ##### 
def Find_Differences_Month_Years(pun, pun2, month):
    od = OrderedDict()
    gen16 = pun2.ix[pun2.index.month == month]
    gen10 = pun.ix[pun.index.month == month]
    od['10-16'] = np.mean((gen16 - gen10)**2)   
    for y in range(2011,2016,1):
        
        



jan1 = pun.ix[pun.index.month == 1 and pun.index.year == 2010]
for y in range(2011, 2016, 1):
   newjan =  pun.ix[pun.index.month == 1 and pun.index.year == 2010]
   print()   
   
   
