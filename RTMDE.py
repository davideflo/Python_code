# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 10:56:20 2017

@author: d_floriello

Real-Time Meteo Data Extractor
"""


#import sys
import requests
from selenium import webdriver
import BeautifulSoup
#import pandas as pd
#import fun_meteo_scraper_82 as ME


#city = sys.argv[1]
#anno = sys.argv[2]
#list_mesi = []
#for arg in sys.argv[3:]:
#  list_mesi.append(arg)

city = ['Milano', 'Firenze', 'Roma', 'Bari', 'Palermo', 'Cagliari']

city = 'Milano'
cromepath = r'C:/Users/d_floriello/Desktop/chromedriver.exe'

browser = webdriver.Chrome(cromepath)

browser.get('http://www.meteo.it')
    
elem = browser.find_element_by_id('searchinput')    
elem.send_keys(city)
elem.submit()

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

gradi = [float(deg[i][0]) for i in range(dagradi+1, dagradi+25)]
vento = [float(deg[i]) for i in range(davento+1, davento+25)]
pioggia = [float(deg[i][0]) for i in range(dapioggia+1, dapioggia+25)]
P = 0
for x in deg:
    if 'Pioggia' in x:
        P = 1
        pass


