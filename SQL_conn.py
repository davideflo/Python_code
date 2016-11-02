# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:22:57 2016

@author: utente

sql connections
"""

import MySQLdb as msql

conn = msql.connect(host = '192.168.0.60', port = 1521, user = 'read_only', password = 'read_only', db = 'T_SWITCH_DETT')
conn = msql.connect(host = '192.168.0.60', port = 1521, user = 'read_only', password = 'read_only')
conn = msql.connect(host = '192.168.0.60', user = 'read_only', password = 'read_only')

import mysql.connector

connection = mysql.connector.connect(user="read_only",passwd="read_only",db="T_SYS_CODIFICA_PARAM",host="192.168.0.60", port=1521)
cur = connection.cursor()

import pyodbc
import mx.ODBC as odbc
#import mx.ODBC.Windows as odbc

cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=192.168.0.60;PORT=1521;DATABASE=T_SYS_CODIFICA_PARAM;UID=read_only;PWD=read_only')

cnxn = pyodbc.connect(driver='{SQL Server}', host='192.168.0.60', port='1521', database='T_SYS_CODIFICA_PARAM',
                      trusted_connection='yes', user='read_only', password='read_only')

cnxn = pyodbc.connect(driver='{ODBC-SQL Server}', host='192.168.0.60', database='T_SYS_CODIFICA_PARAM',
                      trusted_connection='yes', uid='read_only', password='read_only')

dir(odbc)

datasource = odbc.Windows.DataSources

####################################################################################################
def show_odbc_sources():
    sources = pyodbc.dataSources()
    dsns = sources.keys()
    dsns.sort()
    sl = []
    for dsn in dsns:
        sl.append('%s [%s]' % (dsn, sources[dsn]))
        print '\n'.join(sl)
    return sl
####################################################################################################

show_odbc_sources()

