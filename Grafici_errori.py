# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:10:42 2017

@author: d_floriello


"""

import pandas as pd
import matplotlib.pyplot as plt


dati = pd.read_excel("Aggregato_copia.xlsx", sheetname = "Delta LP", skiprows = [0,1])


errore_nord = dati[['Ora', "NORD.2"]].groupby('Ora')

errore_nord.boxplot(by = 'Ora')

errore_csud = dati[['Ora', "CSUD.2"]].groupby('Ora')
errore_csud.boxplot(by = 'Ora')

dati[['Giorno settimana', "CSUD.2"]].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')

dati[['Giorno settimana', "CSUD.2"]].ix[dati['Mese'] == 1].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')

dati[['Giorno settimana', "CNOR.2"]].ix[dati['Mese'] == 2].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')
dati[['Giorno settimana', "NORD.2"]].ix[dati['Mese'] == 2].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')

dati[['Giorno settimana', "SUD.2"]].ix[dati['Mese'] == 2].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')
dati[['Giorno settimana', "SICI.2"]].ix[dati['Mese'] == 2].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')

axes = dati[['Giorno settimana', "SARD.2"]].ix[dati['Mese'] == 2].groupby('Giorno settimana').boxplot(by = 'Giorno settimana').suptitle('Distribuzione errori orari SARD')
fig = axes.get_figure()
fig.suptitle('Distribuzione errori orari SARD')



plt.figure()
dati[['Giorno settimana', "SARD.2"]].ix[dati['Mese'] == 2].groupby('Giorno settimana').boxplot(by = 'Giorno settimana')
plt.title('Distribuzione errori orari SARD')

plt.figure()
plt.boxplot([dati["SARD.2"].ix[dati['Mese'] == 2], dati["SARD.2"].ix[dati['Mese'] == 1]])
###############################################################################
def drawBOXPLOTs(zona):
    plt.figure()
    plt.boxplot([dati[zona + ".2"].ix[dati['Mese'] == 2], dati[zona + ".2"].ix[dati['Mese'] == 1]])
    plt.title('Errori sul forecast orario per mese in {}'.format(zona))
    plt.savefig('H:/Energy Management/20. Strutture blocchi forecast/Grafici Errori/' + zona + '_mese.png')
    ll = []
    for i in range(24):
        ll.append(dati[zona + ".2"].ix[dati['Ora'] == i])
    plt.figure()
    plt.boxplot(ll)
    plt.title('Errori sul forecast orario per ora in {}'.format(zona))
    plt.savefig('H:/Energy Management/20. Strutture blocchi forecast/Grafici Errori/' + zona + '_ora.png')
    ll2 = []
    for i in range(1,8,1):
        ll2.append(dati[zona + ".2"].ix[dati['Giorno settimana'] == i])
    plt.figure()
    plt.boxplot(ll2)
    plt.title('Errori sul forecast orario per giorno in {}'.format(zona))
    plt.savefig('H:/Energy Management/20. Strutture blocchi forecast/Grafici Errori/' + zona + '_giorno.png')
###############################################################################
drawBOXPLOTs("NORD")
drawBOXPLOTs("CNOR")
drawBOXPLOTs("CSUD")
drawBOXPLOTs("SUD")
drawBOXPLOTs("SICI")
drawBOXPLOTs("SARD")
###############################################################################
def drawBOXPLOTsLP(zona):
    plt.figure()
    plt.boxplot([dati[zona + ".2"].ix[dati['Mese'] == 2], dati[zona + ".2"].ix[dati['Mese'] == 1]])
    plt.title('Errori sul forecast LP per mese in {}'.format(zona))
    plt.savefig('H:/Energy Management/20. Strutture blocchi forecast/Backcheck errori/Grafici Errori/' + zona + '_mese.png')
    ll = []
    for i in range(24):
        ll.append(dati[zona + ".2"].ix[dati['Ora'] == i])
    plt.figure()
    plt.boxplot(ll)
    plt.title('Errori sul forecast LP per ora in {}'.format(zona))
    plt.savefig('H:/Energy Management/20. Strutture blocchi forecast/Backcheck errori/Grafici Errori/' + zona + '_ora.png')
    ll2 = []
    for i in range(1,8,1):
        ll2.append(dati[zona + ".2"].ix[dati['Giorno settimana'] == i])
    plt.figure()
    plt.boxplot(ll2)
    plt.title('Errori sul forecast LP per giorno in {}'.format(zona))
    plt.savefig('H:/Energy Management/20. Strutture blocchi forecast/Backcheck errori/Grafici Errori/' + zona + '_giorno.png')
###############################################################################
drawBOXPLOTsLP("NORD")
drawBOXPLOTsLP("CNOR")
drawBOXPLOTsLP("CSUD")
drawBOXPLOTsLP("SUD")
drawBOXPLOTsLP("SICI")
drawBOXPLOTsLP("SARD")