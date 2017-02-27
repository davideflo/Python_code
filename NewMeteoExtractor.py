# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:46:50 2017

@author: utente

New scheduled meteo extractor 
"""

#import sys
import os
import requests
from selenium import webdriver
import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict

#### Ideally, I could run the meteo updating when calling EE.py on the 69. See:
#### http://stackoverflow.com/questions/3781851/run-a-python-script-from-another-python-script-passing-in-args
#### For weekend calls, set up a scheduler. See:
#### https://docs.python.org/2/library/sched.html
#### Or the second answer here:
#### http://stackoverflow.com/questions/373335/how-do-i-get-a-cron-like-scheduler-in-python


def Extractor():
    cities = ['Milano', 'Firenze', 'Roma', 'Bari', 'Palermo', 'Cagliari']
    
    for city in cities:
        
        
        path = 'C:/Users/utente/Documents/meteo'
        cromepath = r'C:/Users/utente/Desktop/chromedriver/chromedriver.exe'
        
        browser = webdriver.Chrome(cromepath)
        
        target_url = 'http://www.meteo.it/meteo/' + city + '-domani-15146'
        
        browser.get(target_url)
            
        
        c_url = browser.current_url        
        r = requests.get(c_url)
        data = r.text        
                
        data_soup = BeautifulSoup.BeautifulSoup(data)        
                
        div = data_soup.findAll("div")        
        div = div[1:]        
                
        deg = [element.text for element in data_soup.findAll("div", "pk_bvalign")]
        
        dagradi = deg.index('Temp.&deg;C&deg;F')
        davento = deg.index('VentoKm/hMph Nodi')
        dapioggia = deg.index('Precipit.mm/cminches')
        
        gradi = [float(deg[i][0].replace(',','.')) for i in range(dagradi+1, dagradi+25)]
        vento = [float(deg[i].replace(',','.')) for i in range(davento+1, davento+25)]
        pioggia = [float(deg[i][:-2].replace(',','.')) for i in range(dapioggia+1, dapioggia+25)]
        
        P = 0
        for x in deg:
            if 'Pioggia' in x:
                P = 1
                pass
        
        dom = OrderedDict()
        dom['Tmin'] = [np.min(gradi)]
        dom['Tmax'] = [np.max(gradi)]
        dom['Tmedia'] = [np.mean(gradi)]
        dom['vento'] = [np.mean(vento)]
        dom['pioggia'] = [P]
        dom['mmpioggia'] = [np.mean(pioggia)]
    
        domani = pd.DataFrame.from_dict(dom, orient = 'columns')
        
        domani.index = pd.date_range(datetime.datetime.today() + datetime.timedelta(days=1), datetime.datetime.today() + datetime.timedelta(days=1))
        
        path2 = path + '/' + city + '.xlsx'
        
        
        df = pd.read_excel(path2)
        df = df.append(domani)
        df.to_excel(path2)
        
        print '{} DONE'.format(city)
        
    return 1