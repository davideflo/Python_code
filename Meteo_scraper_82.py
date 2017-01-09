# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 10:04:02 2017

@author: d_floriello

meteo web scraper server 82

"""

import sys
from selenium import webdriver
#import pandas as pd
import fun_meteo_scraper_82 as ME


city = sys.argv[1]
anno = sys.argv[2]
list_mesi = []
for arg in sys.argv[3:]:
  list_mesi.append(arg)

#browser = webdriver.Firefox()
cromepath = r'C:/Users/d_floriello/Desktop/chromedriver.exe'

browser = webdriver.Chrome(cromepath)

ME.GetListMonth(browser, city, list_mesi, anno)
ME.ElaborateExcel(city, list_mesi)
#city = 'Milano'
#list_mesi = ['Settembre']

###### meteo part

#browser.get('http://www.ilmeteo.it/portale/archivio-meteo')
#
#elem = browser.find_element_by_id('edit-zearch')
#
#elem.send_keys(city)
#
#clicker = browser.find_element_by_name('search_submit')
#clicker.click()
#
#### for i in list_mesi:
#lm = list_mesi[0]
#
#elem2 = browser.find_element_by_link_text(lm)
#elem2.click()
#
#elem3 = browser.find_element_by_xpath('//*[@title="dati storici Milano Settembre 2016"]')
#elem3.click()
#
####### excel part
#
#df = pd.read_excel('C:/Users/d_floriello/Documenti/PUN/'+city+' 2016.xlsx')
#dfnames = df.columns
#
#for m in list_mesi:
#    dfloc = pd.read_csv('C:/Users/d_floriello/Downloads/'+city+'-2016-'+m+'.csv', sep = ';', header = None)
#    dfloc = dfloc.ix[1:]
#    dfloc.columns = dfnames
#    dfloc[dfloc.columns[2:13]] = dfloc[dfloc.columns[2:13]].apply(pd.to_numeric) 
#    df = df.append(dfloc)
#    