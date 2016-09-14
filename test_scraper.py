# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:51:23 2016

@author: d_floriello

data scraper -- test
"""

from selenium import webdriver
browser = webdriver.Firefox()

type(browser)
browser.get('http://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_StimeFabbisogno')

elem_inizio = browser.find_element_by_id('ContentPlaceHolder1_tbDataStart')
elem_fine = browser.find_element_by_id('ContentPlaceHolder1_tbDataStop')

elem_inizio.send_keys('01/01/2016')
elem_fine.send_keys('31/01/2016')
#elem_fine.submit()

clicker = browser.find_element_by_id('ContentPlaceHolder1_btnScarica')
clicker.click()