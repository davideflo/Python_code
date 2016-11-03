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

cnxn = pyodbc.connect(driver='{SQL Server}', host='192.168.0.60', database='argon',
                      trusted_connection='yes', uid='read_only', password='read_only')

cnxn = pyodbc.connect(driver='{Oracle in Oradb11g_home1}', host='192.168.0.60', database='ARGON',
                      trusted_connection='yes', uid='read_only', password='read_only')

connectString = 'Driver={Oracle in Oradb11g_home1};Server=pippo:1521/argon.pippo;uid= read_only;pwd=read_only'
cnxn = pyodbc.connect(connectString)
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

connectString = 'Driver={Microdsoft ODBC for Oracle};Server=192.168.0.60:1521/argon.192.168.0.60;uid= read_only;pwd=read_only'
cnxn = pyodbc.connect(connectString)

####################################################################################################
import os
import cx_Oracle
import csv
 
SQL="SELECT * FROM SOME_TABLE"
 
# Network drive somewhere
filename="S:\Output.csv"
FILE=open(filename,"w");
output=csv.writer(FILE, dialect='excel')
 
# You can set these in system variables but just in case you didnt
os.putenv('ORACLE_HOME', '/oracle/product/10.2.0/db_1') 
os.putenv('LD_LIBRARY_PATH', '/oracle/product/10.2.0/db_1/lib') 
 
dns = cx_Oracle.makedsn('192.168.0.60', '1521', 'argon')
connection = cx_Oracle.connect('read_only', 'read_only', dns) 
cursor = connection.cursor()

SQL = "SELECT * FROM MNH_BILLING.AUDIT_MISURE_GAS_CAMBI_STATO"

cursor.execute(SQL)

for row in cursor:
    print row
    
cursor.close()
connection.close()
FILE.close()



db = cx_Oracle.connect('hr', 'hrpwd', 'localhost:1521/XE')
db1 = cx_Oracle.connect('hr/hrpwd@localhost:1521/XE')
dsn_tns = cx_Oracle.makedsn('localhost', 1521, 'XE')
print dsn_tns
'(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=localhost)(PORT=1521)))(CONNECT_DATA=(SID=XE)))'
db2 = cx_Oracle.connect('hr', 'hrpwd', dsn_tns)
 




