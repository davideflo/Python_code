# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:29:55 2017

@author: d_floriello

automatic download and unzipping of files
"""


#from robobrowser import RoboBrowser
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


#browser = RoboBrowser()
#login_url = 'https://smistaweb.enel.it/tpauth/JavaNotEnabled.html'
#browser.open(login_url)
#form = browser.get_form(id='form_id')

#Profilo: Z:\Lorenzo\Entrust Profile\GABRIELE BERTHOLET.epf
#Password: Axopower_123

#form['profile'].value = "Z:\Lorenzo\Entrust Profile\GABRIELE BERTHOLET.epf" 
#form['password'].value = "Axopower_123"
#browser.submit_form(form)



binary = FirefoxBinary("C:/Program Files (x86)/Mozilla Firefox/firefox.exe")

login_url = 'https://smistaweb.enel.it/tpauth/AuthenticateUserLocalEPF.html'

browser = webdriver.Firefox(firefox_binary = binary)
browser.get(login_url)


page = browser.page_source
##### http://stackoverflow.com/questions/32528871/selenium-source-missing-login-fields
##### pretty much same error here ####
#### SWITCH TO FRAME:
browser.switch_to.frame(browser.find_element_by_name('entrustTruePassGuiFrame'))
profilo = browser.find_element_by_name('username')
pw = browser.find_element_by_name('password')

profilo.send_keys("Z:\Lorenzo\Entrust Profile\GABRIELE BERTHOLET.epf")
pw.send_keys("Axopower_123")

login = browser.find_element_by_xpath("//input[@value='Login']")
login.click()

browser_ie = webdriver.Ie('C:/Program Files/Internet Explorer/iexplore.exe')
browser_ie.get(login_url)