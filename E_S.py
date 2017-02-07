# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 15:29:35 2017

@author: d_floriello

enel scraper
"""

from robobrowser import RoboBrowser
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

browser = RoboBrowser()
login_url = 'https://smistaweb.enel.it/tpauth/JavaNotEnabled.html'
browser.open(login_url)
form = browser.get_form(id='form_id')

browser = webdriver.Firefox()
browser.get(login_url)

#Profilo: Z:\Lorenzo\Entrust Profile\GABRIELE BERTHOLET.epf
#Password: Axopower_123

form['profile'].value = "Z:\Lorenzo\Entrust Profile\GABRIELE BERTHOLET.epf" 
form['password'].value = "Axopower_123"
browser.submit_form(form)



binary = FirefoxBinary("C:/Program Files (x86)/Mozilla Firefox/firefox.exe")

login_url = 'https://smistaweb.enel.it/tpauth/JavaNotEnabled.html'

browser = webdriver.Firefox(firefox_binary = binary)
browser.get(login_url)