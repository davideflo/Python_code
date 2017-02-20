# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 10:56:20 2017

@author: d_floriello

Real-Time Meteo Data Extractor
"""


#import sys
import os
import requests
from selenium import webdriver
import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
#import fun_meteo_scraper_82 as ME

#### Ideally, I could run the meteo updating when calling EE.py on the 69. See:
#### http://stackoverflow.com/questions/3781851/run-a-python-script-from-another-python-script-passing-in-args
#### For weekend calls, set up a scheduler. See:
#### https://docs.python.org/2/library/sched.html
#### Or the second answer here:
#### http://stackoverflow.com/questions/373335/how-do-i-get-a-cron-like-scheduler-in-python


#city = sys.argv[1]
#anno = sys.argv[2]
#list_mesi = []
#for arg in sys.argv[3:]:
#  list_mesi.append(arg)

city = ['Milano', 'Firenze', 'Roma', 'Bari', 'Palermo', 'Cagliari']

city = 'Milano'

path = 'C:/Users/utente/Documents/meteo'
cromepath = r'C:/Users/d_floriello/Desktop/chromedriver.exe'

browser = webdriver.Chrome(cromepath)

target_url = 'http://www.meteo.it/meteo/' + city + '-domani-15146'

browser.get(target_url)
    
#elem = browser.find_element_by_id('searchinput')    
#elem.send_keys(city)
#elem.submit()
#
#browser.find_element_by_class_name('title').click()

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
pioggia = [float(deg[i][0].replace(',','.')) for i in range(dapioggia+1, dapioggia+25)]

P = 0
for x in deg:
    if 'Pioggia' in x:
        P = 1
        pass

domani = pd.DataFrame.from_dict({'Tmin': np.min(gradi), 'Tmax': np.max(gradi), 'Tmedia': np.mean(gradi),
                                 'vento': np.mean(vento), 'pioggia': P})
domani.set_index(datetime.datetime.now().date())

if os.path.isfile(path):