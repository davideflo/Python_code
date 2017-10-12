# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:19:40 2017

@author: utente

Sbilanciamento 15 -- Estimation of Temperature dependence models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy
from sklearn.externals import joblib



####################################################################################################
def AddHolidaysDate(vd):
    
  ##### codifica numerica delle vacanze
  ## 1 Gennaio = 1, Epifania = 2
  ## Pasqua = 3, Pasquetta = 4
  ## 25 Aprile = 5, 1 Maggio = 6, 2 Giugno = 7,
  ## Ferragosto = 8, 1 Novembre = 9
  ## 8 Dicembre = 10, Natale = 11, S.Stefano = 12, S.Silvestro = 13
    holidays = 0
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16)]

    if vd.month == 1 and vd.day == 1:
        holidays = 1
    if vd.month  == 1 and vd.day == 6: 
        holidays = 1
    if vd.month  == 4 and vd.day == 25: 
        holidays = 1
    if vd.month  == 5 and vd.day == 1: 
        holidays = 1
    if vd.month  == 6 and vd.day == 2: 
        holidays = 1
    if vd.month  == 8 and vd.day == 15: 
        holidays = 1
    if vd.month  == 11 and vd.day == 1: 
        holidays = 1
    if vd.month  == 12 and vd.day == 8: 
        holidays = 1
    if vd.month  == 12 and vd.day == 25: 
        holidays = 1
    if vd.month  == 12 and vd.day == 26: 
        holidays = 1
    if vd.month  == 12 and vd.day == 31: 
        holidays = 1
    if vd in pasqua:
        holidays = 1
    if vd in pasquetta:
        holidays = 1
  
    return holidays
####################################################################################################
def Get_MeanDependencyWithTemperature(df, meteo):
### @PARAM: df is the dataset I want to compute the dependency of: it could be Sample, OOS or Zonal and
### needs to be a time series
    
#    final = min(max(df.index.date), max(meteo.index.date))
    final = min(max(df.index.date), max(meteo.index))
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    
    #di = max(min(df.index.date), min(meteo.index.date))  
    di = max(min(df.index.date), min(meteo.index)) 
    #basal_cons = min(df.ix[df.index.month == 4].mean().values.ravel()[0] ,df.ix[df.index.month == 5].mean().values.ravel()[0])   
    basal_cons = df.mean().values[0]
    
    dts = OrderedDict()
    indices = pd.date_range(di, final_date, freq = 'D')
    for i in indices:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        dvector[wd] = 1
        mvector[(i.month-1)] = 1
        dy = i.timetuple().tm_yday
#        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
#        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
#        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        Tmax = meteo['Tmax'].ix[meteo.index == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index == i.date()].values.ravel()[0]        
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())

        y = df.ix[i].mean()
        ### detrend by the mean?               
        
        ll.append(y - basal_cons)        
        dts[i] =  ll
        
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls','y']]
    return dts
####################################################################################################
def GetPastSample2(db, rical, sos, pdo, som, year, zona):
    
    current_month = datetime.datetime.now().month
    PP = pd.DataFrame()
    missing = []
    strm = str(current_month) if len(str(current_month)) > 1 else "0" + str(current_month)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])    
    db = db.ix[db['Area'] == zona]
    podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
    kc = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

    ricpod = []    
    if len(som) == 12:
        for p in podlist:
            if p in rical.columns:
                ricpod.append(p)
    
    cols = ['Giorno', 'Ora']
    cols.extend(ricpod)
    RIC = rical[cols]    
    
    podlist = list(set(podlist).difference(set(ricpod)))
    for s in som:
        for p in podlist:        
            pdop = pdo.ix[pdo['POD'] == p]
            pdop = pdop.reset_index(drop = True)
            pdopm = pdop.ix[np.where(np.array( map(lambda date: date.month, pdop['Giorno'].values.ravel()) ) == s)]
            pdopm = pdopm.reset_index(drop = True)
            pdopmy = pdopm.ix[np.where(np.array( map(lambda date: date.year, pdopm['Giorno'].values.ravel()) ) == year)]
            pdopmy = pdopmy.drop_duplicates(subset = ['POD', 'Giorno'], keep = 'last')
            pdopmy = pdopmy[pdopmy.columns[kc]]
            if pdopmy.shape[0] > 0:            
                PP = PP.append(pdopmy, ignore_index = True)
            else:
                sosp = sos.ix[sos['Pod'] == p]
                sosp = sosp.reset_index(drop = True)
                sospm = sosp.ix[np.where(np.array( map(lambda date: date.month, sosp['Giorno'].values.ravel())) == s)]
                sospm = sospm.reset_index(drop = True)
                #sospmy = sospm.ix[np.where(np.array( map(lambda date: date.year, sospm['Giorno'].values.ravel())) == year)]
                sospm = sospm.drop_duplicates(subset = ['Pod', 'Giorno'], keep = 'last')
                #sospm = sospm[sospm.columns[kc]]
                if sospm.shape[0] > 0:
                    PP = PP.append(sospm, ignore_index = True)
                else:
                    missing.append((p, s))
    
    if PP.shape[0] > 0:    
        PP = PP.set_index(pd.to_datetime(PP['Giorno']))
        PP = PP.resample('D').sum()/1000    
        print 'missing {} PODs'.format(len(missing))       
        return PP, missing, RIC
        
    else:
        return RIC, missing
####################################################################################################
def GetWeatherModelAuto():
    zone = ['NORD', 'CNOR', 'CSUD', 'SUD', 'SICI', 'SARD']
    cities = ['Milano', 'Firenze', 'Fiumicino', 'Bari', 'Palermo', 'Cagliari']
    som = range(1,13)
    sos = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.xlsx")
    pdo = pd.read_hdf("C:/Users/utente/Documents/DB_misure.h5")
    
    db = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
    db.columns = [str(i) for i in db.columns]
    db = db[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
             "16","17","18","19","20","21","22","23","24"]]
    db = db.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')
    
    for i in range(7):
        zona = zone[i]
        city = cities[i]
        rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona)    

        me7 = pd.read_excel('C:/Users/utente/Documents/PUN/' + city + ' 2017.xlsx')
        me7 = me7.set_index(pd.date_range('2017-01-01', '2017-09-30', freq = 'D'))
        me7 = me7[['Tmin', 'Tmax','Tmedia', 'VENTOMEDIA', 'PIOGGIA']]
        me7.columns = ["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]
        
        me6 = pd.read_excel('C:/Users/utente/Documents/PUN/' + city + ' 2016.xlsx')
        me6 = me6.drop(59)
        me6 = me6.ix[:365].set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'D'))
        me6 = me6[['Tmin', 'Tmax','Tmedia', 'VENTOMEDIA', 'PIOGGIA']]
        me6.columns = ["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]
        
        me = me7.append(me6.ix[me6.index.month > 9])
        me = me.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'D').date)
        
        ret = GetPastSample2(db, rical, sos, pdo, som, 2017, zona)
        
        df = ret[2]
        df = pd.DataFrame({'S': df.values.ravel()/1000}).set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:df.values.ravel().size])

        DWT = Get_MeanDependencyWithTemperature(df, mi)

        DWTt = DWT.sample(frac = 1)

        DWTtr = DWTt.ix[:int(np.ceil(0.8*DWTt.shape[0]))]
        DWTte = DWTt.ix[int(np.ceil(0.8*DWTt.shape[0])):]
        
        rf2 = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
        rf2.fit(DWTtr[DWTtr.columns[:27]], DWTtr['y'])
        
        print 'R2 train set: {}'.format(r2_score(DWTtr['y'], rf2.predict(DWTtr[DWTtr.columns[:27]])))
        print 'MSE train set: {}'.format(mean_squared_error(DWTtr['y'], rf2.predict(DWTtr[DWTtr.columns[:27]])))
        print 'R2 test set: {}'.format(r2_score(DWTte['y'], rf2.predict(DWTte[DWTte.columns[:27]])))
        print 'MSE test set: {}'.format(mean_squared_error(DWTte['y'], rf2.predict(DWTte[DWTte.columns[:27]])))
        
        resid = DWTte['y'].values.ravel() - rf2.predict(DWTte[DWTte.columns[:27]])

        plt.figure()
        plt.plot(resid)
        plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.95), color = 'k')
        plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.0275), color = 'k')
        plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.75), color = 'green')
        plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.25), color = 'green')
        plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.8), color = 'yellow')
        plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.2), color = 'yellow')
        plt.title('grafico residui con quantili zona' + zona)
        
        plt.figure()
        plt.hist(resid, bins = 20)
        plt.title('istogramma residui')
        
        plt.figure()
        plt.scatter(DWTte['y'].values.ravel(), rf2.predict(DWTte[DWTte.columns[:27]]))
        
        print 'media residui: {}'.format(np.mean(resid))
        print 'mediana residui: {}'.format(np.mean(resid))
        print 'deviazione standard residui: {}'.format(np.std(resid))
        print 'skewness residui: {} (sk < 0 => coda destra pesante)'.format(scipy.stats.skew(resid))
        
        q = scipy.stats.mstats.mquantiles(resid, prob = np.linspace(0.0,1,100))        
        dq = np.diff(q)        
        
        plt.figure()
        plt.plot(dq, color = 'violet', marker = 'o')
        plt.title('differenza tra quantili consecutivi')
        
        first = dq[dq < 0.2][0]        
        last = dq[dq < 0.2][-1]

        first_index = dq.tolist().tolist().index(first)
        last_index = dq.tolist().tolist().index(last)        
        
        print '''percentuale di errori 'accettabile': {}
        '''.format(float(np.sum((resid[first_index + 1] <= resid) * (resid <= resid[last_index + 1])))/float(resid.size))
        
        joblib.dump(rf2, 'C:/Users/utente/Documents/Sbilanciamento/model_weather_S_' + zona + '.pkl')
        
    return 1
####################################################################################################

mi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2017.xlsx')
mi7 = mi7.set_index(pd.date_range('2017-01-01', '2017-09-30', freq = 'D'))
mi7 = mi7[['Tmin', 'Tmax','Tmedia', 'VENTOMEDIA', 'PIOGGIA']]
mi7 = mi7.set_index(np.array(list(map(lambda date: date.date(), pd.to_datetime(mi7.index)))))
mi7.columns = ["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]

mi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2016.xlsx')
mi6 = mi6.drop(59)
mi6 = mi6.ix[:365].set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'D'))
mi6 = mi6[['Tmin', 'Tmax','Tmedia', 'VENTOMEDIA', 'PIOGGIA']]
mi6 = mi6.set_index(np.array(list(map(lambda date: date.date(), pd.to_datetime(mi6.index)))))
mi6.columns = ["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]

mi = mi7.append(mi6.ix[mi6.index.month > 9])
mi = mi.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'D').date)

ret = GetPastSample2(db, rical, sos, pdo, range(1,13), 2017, zona)

df = ret[2]
df = pd.DataFrame({'S': df.values.ravel()/1000}).set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:df.values.ravel().size])

DWT = Get_MeanDependencyWithTemperature(df, mi)

DWTt = DWT.sample(frac = 1)

DWTtr = DWTt.ix[:int(np.ceil(0.8*DWTt.shape[0]))]
DWTte = DWTt.ix[int(np.ceil(0.8*DWTt.shape[0])):]


rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
rf.fit(DWTt[DWTt.columns[:27]], DWTt['y'])

print r2_score(DWTt['y'], rf.predict(DWTt[DWTt.columns[:27]]))
print mean_squared_error(DWTt['y'], rf.predict(DWTt[DWTt.columns[:27]]))


rf2 = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
rf2.fit(DWTtr[DWTtr.columns[:27]], DWTtr['y'])

print r2_score(DWTtr['y'], rf2.predict(DWTtr[DWTtr.columns[:27]]))
print mean_squared_error(DWTtr['y'], rf2.predict(DWTtr[DWTtr.columns[:27]]))
print r2_score(DWTte['y'], rf2.predict(DWTte[DWTte.columns[:27]]))
print mean_squared_error(DWTte['y'], rf2.predict(DWTte[DWTte.columns[:27]]))

plt.figure()
plt.scatter(range(1, DWTte.shape[0]+1), DWTte['y'].values.ravel(), color = 'blue')
plt.scatter(range(1, DWTte.shape[0]+1), rf2.predict(DWTte[DWTte.columns[:27]]), color = 'red', marker = 'o')

resid = DWTte['y'].values.ravel() - rf2.predict(DWTte[DWTte.columns[:27]])

plt.figure()
plt.plot(resid)
plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.95), color = 'k')
plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.0275), color = 'k')
plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.75), color = 'green')
plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.25), color = 'green')
plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.8), color = 'yellow')
plt.axhline(y = scipy.stats.mstats.mquantiles(resid, prob = 0.2), color = 'yellow')

plt.figure()
plt.hist(resid, bins = 20)

plt.figure()
plt.scatter(DWTte['y'].values.ravel(), rf2.predict(DWTte[DWTte.columns[:27]]))

print np.mean(resid)
print np.std(resid)
print scipy.stats.skew(resid)

plt.figure()
plt.plot(rf2.predict(DWTte[DWTte.columns[:27]]))

plt.figure()
plt.plot(np.diff(scipy.stats.mstats.mquantiles(resid, prob = np.linspace(0.05,1,100))), color = 'violet', marker = 'o')

print float(np.sum((-2.74753128 <= resid) * (resid <= 1.64590391)))/float(resid.size)