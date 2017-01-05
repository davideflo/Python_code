# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:45:24 2016

@author: d_floriello

Functions for meteo scraper
"""
import pandas as pd


def GetMonth(browser, city, lm, anno):
    browser.get('http://www.ilmeteo.it/portale/archivio-meteo')
    
    elem = browser.find_element_by_id('edit-zearch')
    
    elem.send_keys(city)
        
    clicker = browser.find_element_by_name('search_submit')
    clicker.click()

    browser.find_element_by_xpath("//select[@name='anno']/option[text()=" + str(anno) + "]").click()        
    
    elem2 = browser.find_element_by_link_text(lm)
    elem2.click()
    
    browser.find_element_by_xpath("//select[@name='anno']/option[text()=" + str(anno) + "]").click()    
    browser.find_element_by_name("conferma").click()
    
    # '//*[@title="dati storici Milano Settembre 2016"]'
    xpath = '//*[@title="dati storici ' + city + ' ' + lm + ' 2016"]'
    elem3 = browser.find_element_by_xpath(xpath)
    elem3.click()
    
    print('done {} in {}'.format(city, lm))
    return 1
###############################################################################
def GetListMonth(browser, city, list_month, anno):
    for lm in list_month:
        GetMonth(browser, city, lm, anno)
    return 1
###############################################################################
def ElaborateExcel(city, list_mesi):
    df = pd.read_excel('C:/Users/d_floriello/Documenti/PUN/'+city+' 2016.xlsx')
    dfnames = df.columns
    
    for m in list_mesi:
        dfloc = pd.read_csv('C:/Users/d_floriello/Downloads/'+city+'-2016-'+m+'.csv', sep = ';', header = None)
        dfloc = dfloc.ix[1:]
        dfloc.columns = dfnames
        dfloc[dfloc.columns[2:13]] = dfloc[dfloc.columns[2:13]].apply(pd.to_numeric) 
        df = df.append(dfloc)
    
    print(df)
    df.to_excel('C:/Users/d_floriello/Documenti/PUN/'+city+' 2016_updated.xlsx')
    return df
###############################################################################