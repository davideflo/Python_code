# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:46:50 2017

@author: utente

New scheduled meteo extractor 
"""

#import sys
import os
#import requests
from selenium import webdriver
#import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict
#from PyQt4.QtGui import *  
#from PyQt4.QtCore import *  
#from PyQt4.QtWebKit import *  
#from lxml import html 
import re
#import time
import pyscreenshot as ImageGrab
from subprocess import Popen, PIPE
#import contexlib

#### Ideally, I could run the meteo updating when calling EE.py on the 69. See:
#### http://stackoverflow.com/questions/3781851/run-a-python-script-from-another-python-script-passing-in-args
#### For weekend calls, set up a scheduler. See:
#### https://docs.python.org/2/library/sched.html
#### Or the second answer here:
#### http://stackoverflow.com/questions/373335/how-do-i-get-a-cron-like-scheduler-in-python

#####################################################################################################
#class Render(QWebPage):  
#  def __init__(self, url):  
#    self.app = QApplication(sys.argv)  
#    QWebPage.__init__(self)  
#    self.loadFinished.connect(self._loadFinished)  
#    self.mainFrame().load(QUrl(url))  
#    self.app.exec_()  
#    self.frame = self.mainFrame()
#  
#  def _loadFinished(self, result):  
#    self.frame = self.mainFrame()  
#    self.app.quit() 
####################################################################################################
def Extractor():
    cities = ['Milano-domani-15146', 'Firenze-domani-48017', 'Roma-domani-58091', 
    'Bari-domani-72006', 'Palermo-domani-82053', 'Cagliari-domani-92009']
    
#    city = cities[0]
    for city in cities:
        
        
        path = 'C:/Users/utente/Documents/meteo'
        cromepath = r'C:/Users/utente/Desktop/chromedriver/chromedriver.exe'
        
        browser = webdriver.Chrome(cromepath)
        
        url = 'http://www.meteo.it/meteo/' + city
        
        browser.get(url)
        
        page = browser.page_source
        result = page
#        start_time = time.time()
### forse si blocca per questo:
# http://stackoverflow.com/questions/21909907/pyqt-class-not-working-for-the-second-usage/21918243#21918243        
#        r = Render(url)  
#        result = r.frame.toHtml()
#        del r
#        print("--- %s seconds ---" % (time.time() - start_time))  

        ### find the temperatures  
        temp = [m.start() for m in re.finditer(";deg;", result)]
        Temp = [float(re.findall(r'\d+', result[temp[i]:temp[i]+12])[0]) for i in range(len(temp))]
        ### find the wind r'Part 1(.*?)Part 3'
        wind = [m.start() for m in re.finditer("pk_cventi", result)]
        Vento2 = [re.findall(r'\d+', result[wind[i]:wind[i]+200])[-2] for i in range(len(wind))]
        Vento1 = ['.' + re.findall(r'\d+', result[wind[i]:wind[i]+150])[-1] for i in range(len(wind))]
        Vento = [float(Vento2[i] + Vento1[i]) for i in range(24)]
        ### find the rain 
        rain = [m.start() for m in re.finditer('mm</span>', result)]
        Pioggia = [float(re.findall(r'\d+', result[rain[i]-5:rain[i]])[0]) for i in range(len(rain))]
        
        print 'Done extracting the values'
        
#        c_url = browser.current_url        
#        r = requests.get(c_url)
#        data = r.text        
#                
#        data_soup = BeautifulSoup.BeautifulSoup(data)        
#                
#        div = data_soup.findAll("div")        
#        li = data_soup.findAll("li")
#        div = div[1:]        
#        
#        tsoup = BeautifulSoup.BeautifulSoup(data_soup.findAll("pk_temp"))        
#        
#        ltemp = []
#        lvento = []
#        lpioggia = []
#        for l in li:
#            if 'pk_bvalign' in str(l):
#                ltemp.append(l.div)
#            elif 'pk_cventi' in str(l):
#                lvento.append(l)
#            elif 'pk_bgocce' in str(l):
#                lpioggia.append(l)
#            else:
#                next
#                
#        deg = [element.text for element in data_soup.findAll("div", "pk_bvalign")]
#        
#        dagradi = deg.index('Temp.&deg;C&deg;F')
#        davento = deg.index('VentoKm/hMph Nodi')
#        dapioggia = deg.index('Precipit.mm/cminches')
#        
#        gradi = [float(deg[i][:deg[i].find('&')].replace(',','.')) for i in range(dagradi+1, dagradi+25)]
#        vento = [float(deg[i].replace(',','.')) for i in range(davento+1, davento+25)]
#        pioggia = [float(deg[i][:-2].replace(',','.')) for i in range(dapioggia+1, dapioggia+25)]
#        
        P = 0
        if np.sum(Pioggia) > 0:
            P = 1
            
        
        dom = OrderedDict()
        dom['Tmin'] = [np.min(Temp)]
        dom['Tmax'] = [np.max(Temp)]
        dom['Tmedia'] = [np.mean(Temp)]
        dom['vento'] = [np.mean(Vento)]
        dom['pioggia'] = [P]
        dom['mmpioggia'] = [np.sum(Pioggia)]
    
        domani = pd.DataFrame.from_dict(dom, orient = 'columns')
        
        domani.index = pd.date_range(datetime.datetime.today() + datetime.timedelta(days=1), datetime.datetime.today() + datetime.timedelta(days=1))
        
        path2 = path + '/' + city[:city.find('-')] + '.xlsx'
        
        
        df = pd.read_excel(path2)
        df = df.append(domani)
        df.to_excel(path2)
        
        print '{} DONE'.format(city)
        
        browser.quit()
        
    return 1
####################################################################################################
def ExtractorMoreDays():
    
    cod = ["dopodomani", "3-giorni", "4-giorni", "5-giorni", "6-giorni"]    
    cod_num = ["-15146", "-48017", "-58091", "-72006", "-82053", "-92009"]    
    
    cities = ['Milano', 'Firenze', 'Roma', 'Bari', 'Palermo', 'Cagliari']
    
#    city = cities[0]
    for c in cod:
        for j in range(len(cities)):
            
            
            path = 'C:/Users/utente/Documents/meteo'
            cromepath = r'C:/Users/utente/Desktop/chromedriver/chromedriver.exe'
            
            browser = webdriver.Chrome(cromepath)
            
            url = 'http://www.meteo.it/meteo/' + cities[j] + "-" + c + cod_num[j]
            
            browser.get(url)
            
            page = browser.page_source
            result = page
    
            ### find the temperatures  
            temp = [m.start() for m in re.finditer(";deg;", result)]
            Temp = [float(re.findall(r'\d+', result[temp[i]:temp[i]+12])[0]) for i in range(len(temp))]
            ### find the wind r'Part 1(.*?)Part 3'
            wind = [m.start() for m in re.finditer("pk_cventi", result)]
            Vento2 = [re.findall(r'\d+', result[wind[i]:wind[i]+200])[-2] for i in range(len(wind))]
            Vento1 = ['.' + re.findall(r'\d+', result[wind[i]:wind[i]+150])[-1] for i in range(len(wind))]
            Vento = [float(Vento2[i] + Vento1[i]) for i in range(5)]
            ### find the rain 
            rain = [m.start() for m in re.finditer('mm</span>', result)]
            Pioggia = [float(re.findall(r'\d+', result[rain[i]-5:rain[i]])[0]) for i in range(len(rain))]
            
            print 'Done extracting the values'
            
            P = 0
            if np.sum(Pioggia) > 0:
                P = 1            
            
            dom = OrderedDict()
            dom['Tmin'] = [np.min(Temp)]
            dom['Tmax'] = [np.max(Temp)]
            dom['Tmedia'] = [np.mean(Temp)]
            dom['vento'] = [np.mean(Vento)]
            dom['pioggia'] = [P]
            dom['mmpioggia'] = [np.sum(Pioggia)]
        
            domani = pd.DataFrame.from_dict(dom, orient = 'columns')
            
            DELTA = cod.index(c) + 2            
            
            domani.index = pd.date_range(datetime.datetime.today() + datetime.timedelta(days = DELTA), datetime.datetime.today() + datetime.timedelta(days = DELTA))
            
            path2 = path + '/' + c + "/" + cities[j] + '.xlsx'
            
            if os.path.isfile(path2):
                df = pd.read_excel(path2)
                df = df.append(domani)
                df.to_excel(path2)
            else:
                domani.to_excel(path2)    
                
            print '{} DONE'.format(cities[j] + "-" + c)
            browser.quit()
        
    return 1
####################################################################################################
#def getImage():
#    
#    path = 'C:/Users/utente/Downloads/Terna screenshot/'
#    cromepath = r'C:/Users/utente/Desktop/chromedriver/chromedriver.exe'
#    alt_cromepath = r'C:/Users/utente/Downloads/chromedriver_win32/chromedriver.exe'
#        
#    browser = webdriver.Chrome(cromepath)
#    browser = webdriver.Chrome(alt_cromepath)    
#    
#    url = "http://www.terna.it/DesktopModules/GraficoTerna/GraficoTernaEsteso/ctlGraficoTerna.aspx?sLang=it-IT"
#    
#    browser.get(url)
#    im = browser.get_screenshot_as_png()
#    browser.save_screenshot(path + 'curva' + str(datetime.datetime.now()) + '.png')  
#    browser.get_screenshot_as_file(path + 'curva' + str(datetime.datetime.now()) + '.png')
#    
#    im = ImageGrab.grab()
#    
#    im.save(path + datetime.datetime.now())
#    
#    im.show()
#####################################################################################################
#
#path = 'C:/Users/utente/Downloads/Terna screenshot/'
#abspath = lambda *p: os.path.abspath(os.path.join(*p))
#ROOT = abspath(os.path.dirname(__file__))
#
#
#def execute_command(command):
#    result = Popen(command, shell=True, stdout=PIPE).stdout.read()
#    if len(result) > 0 and not result.isspace():
#        raise Exception(result)
#
#
#def do_screen_capturing(url, screen_path, width, height):
#    print "Capturing screen.."
#    driver = webdriver.PhantomJS("C:/Users/utente/Downloads/phantomjs-2.1.1-windows/bin/phantomjs.exe")
#    # it save service log file in same directory
#    # if you want to have log file stored else where
#    # initialize the webdriver.PhantomJS() as
#    # driver = webdriver.PhantomJS(service_log_path='/var/log/phantomjs/ghostdriver.log')
#    driver.set_script_timeout(30)
#    if width and height:
#        driver.set_window_size(width, height)
#    driver.get(url)
#    driver.save_screenshot(screen_path)
#
#
#def get_screen_shot(**kwargs):
#    url = kwargs['url']
#    #width = int(kwargs.get('width', 1024)) # screen width to capture
#    #height = int(kwargs.get('height', 768)) # screen height to capture
#    filename = kwargs.get('filename', 'screen.png') # file name e.g. screen.png
#    path = kwargs.get('path', 'C:/Users/utente/Downloads/Terna screenshot/') # directory path to store screen
#
#    screen_path = abspath(path, filename)
#
#    do_screen_capturing(url, screen_path, width=1024, height=768)
#
#    return screen_path
#
#
#if __name__ == '__main__':
#    '''
#        Requirements:
#        Install NodeJS
#        Using Node's package manager install phantomjs: npm -g install phantomjs
#        install selenium (in your virtualenv, if you are using that)
#        install imageMagick
#        add phantomjs to system path (on windows)
#    '''
#
#    url = "http://www.terna.it/DesktopModules/GraficoTerna/GraficoTernaEsteso/ctlGraficoTerna.aspx?sLang=it-IT"
#    screen_path = get_screen_shot(
#        url=url, filename='sof' + str(datetime.datetime.now()) + '.png',
#        crop=False, crop_replace=False,
#        thumbnail=False, thumbnail_replace=False,
#        thumbnail_width=0, thumbnail_height=0)