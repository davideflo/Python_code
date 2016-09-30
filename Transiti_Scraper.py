# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 08:47:22 2016

@author: d_floriello

Transiti scraper
"""

from selenium import webdriver
import time
browser = webdriver.Firefox()

type(browser)

date_list = ['01/01/2016', '31/01/2016', '01/02/2016', '29/02/2016', '01/03/2016', '31/03/2016',
             '01/04/2016', '30/04/2016', '01/05/2016', '31/05/2016', '01/06/2016', '30/06/2016',
             '01/07/2016', '31/07/2016', '01/08/2016', '31/08/2016', '01/09/2016', '29/09/2016']

for i in range(0, len(date_list), 2):
    if i == 0:
        browser.get('http://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Transiti')
    
        tick_cont = browser.find_element_by_id('ContentPlaceHolder1_CBAccetto1')
        tick_cont.click()
        tick_acc = browser.find_element_by_id('ContentPlaceHolder1_CBAccetto2')
        tick_acc.click()
        
        acc = browser.find_element_by_id('ContentPlaceHolder1_Button1')
        acc.click()
        
        elem_inizio = browser.find_element_by_id('ContentPlaceHolder1_tbDataStart')
        elem_fine = browser.find_element_by_id('ContentPlaceHolder1_tbDataStop')
        
        elem_inizio.send_keys(date_list[i])
        elem_fine.send_keys(date_list[i+1])
        time.sleep(5)
        clicker = browser.find_element_by_id('ContentPlaceHolder1_btnScarica')
        clicker.click()
    else:
        elem_inizio.clear()
        elem_fine.clear()
        
        elem_inizio.send_keys(date_list[i])
        elem_fine.send_keys(date_list[i+1])
        time.sleep(5)
        clicker = browser.find_element_by_id('ContentPlaceHolder1_btnScarica')
        clicker.click()
