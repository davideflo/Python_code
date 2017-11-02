# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 11:24:44 2017

@author: utente

Sequential Portfolio Analysis -- Sbilanciamento 13
"""

from __future__ import division
import pandas as pd
import numpy as np
import datetime

####################################################################################################
def PortfolioDifference(month1, month2, zona):

    strm1 = str(month1) if len(str(month1)) > 1 else "0" + str(month1)
    crpp1 = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm1 + '-2017/_All_CRPP_' + strm1 + '_2017.xlsx')
    strm2 = str(month2) if len(str(month2)) > 1 else "0" + str(month2)
    crpp2 = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm2 + '-2017/_All_CRPP_' + strm2 + '_2017.xlsx')
    
    crppz1 = crpp1.ix[crpp1['ZONA'] == zona]
    crppz2 = crpp2.ix[crpp2['ZONA'] == zona]
    
    crppz1 = crppz1.ix[crppz1['Trattamento_' + strm1] == 'O']
    crppz2 = crppz2.ix[crppz2['Trattamento_' + strm2] == 'O']
    
    podmonth1 = list(set(crppz1['POD'].values.ravel().tolist()))    
    podmonth2 = list(set(crppz2['POD'].values.ravel().tolist()))    

    exiting = list(set(podmonth1).difference(set(podmonth2)))
    entering = list(set(podmonth2).difference(set(podmonth1)))

    print '{} PODs left us'.format(len(exiting))
    print '{} new PODs with us'.format(len(entering))

#    ml = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#    crppa1 = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/CRPP_' + ml[month1 - 2] + '_2017_artigianale.xlsx') 
#    crppa2 = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/CRPP_' + ml[month2 - 2] + '_2017_artigianale.xlsx')     
    
    Tot1 = 0; Exit = 0    
    Tot2 = 0; Enter = 0
    
    for p in podmonth1:
        if crppz1.ix[crppz1['POD'] == p].shape[0] > 0:
            Tot1 += crppz1['CONSUMO_TOT'].ix[crppz1['POD'] == p].values.ravel()[0]
            if p in exiting:
                Exit += crppz1['CONSUMO_TOT'].ix[crppz1['POD'] == p].values.ravel()[0]
    
    for p in podmonth2:
        if crppz2.ix[crppz2['POD'] == p].shape[0] > 0:
            Tot2 += crppz2['CONSUMO_TOT'].ix[crppz2['POD'] == p].values.ravel()[0]
            if p in entering:
                Enter += crppz2['CONSUMO_TOT'].ix[crppz2['POD'] == p].values.ravel()[0]
    
    print 'TOT out volumes: {} MW p/a'.format(Exit/1000)
    print 'TOT out volumes in terms of zonal weight on the month: {} MW p/a'.format((Exit/1000)/(Tot1/1000))
    print 'TOT new volumes: {} MW p/a'.format(Enter/1000)
    print 'TOT out volumes in terms of zonal weight on the month: {} MW p/a'.format((Enter/1000)/(Tot2/1000))
    print 'NET new zonal consumpion: {} MW p/a'.format((Tot2 - Tot1)/1000)
    
    return Tot1/1000, Exit/1000, Tot2/1000, Enter/1000
####################################################################################################
#def GetBasalConsumption(zona):
#    
#    dm = datetime.datetime.now().month
#    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
#    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
#    crpp_h = crpp.ix[crpp['Trattamento_' + strm] == 'O']    
#    crpp_h = crpp_h.ix[crpp_h.ZONA == zona]    
#    
#    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])