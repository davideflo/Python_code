# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:22:57 2016

@author: utente

sql connections
"""
####################################################################################################
#import os
from __future__ import division
import cx_Oracle
import csv
from collections import OrderedDict
 
SQL="SELECT * FROM SOME_TABLE"
 
# Network drive somewhere
filename="S:\Output.csv"
FILE=open(filename,"w");
output=csv.writer(FILE, dialect='excel')
 
# You can set these in system variables but just in case you didnt
#os.putenv('ORACLE_HOME', '/oracle/product/10.2.0/db_1') 
#os.putenv('LD_LIBRARY_PATH', '/oracle/product/10.2.0/db_1/lib') 
 
dns = cx_Oracle.makedsn('192.168.0.60', '1521', 'argon')
connection = cx_Oracle.connect('read_only', 'read_only', dns) 
cursor = connection.cursor()

SQL = "SELECT * FROM MNH_BILLING.AUDIT_MISURE_GAS_CAMBI_STATO"
sql_all = 'SELECT owner,table_name FROM dba_tables'

cursor.execute(SQL)
cursor.execute(sql_all)

own = OrderedDict()
macros = []

for row in cursor:
    print row
    macros.append(row[0])    

macro = list(set(macros))
### number of macro categories in the database -- owner
len(macro)    
    
call = cursor.execute(sql_all)

for row in call:
    title = row[0]
    if title in own.keys():
        own[title].append(row[1])
    else:
        own[title] = ['AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA']
        own[title].append(row[1])        

#### total number of tables, including the SYS ones
counter = 0    
for k in own.keys():
    counter += len(own[k])
    
import xlsxwriter
workbook = xlsxwriter.Workbook('tables.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0

order=own.keys()
for key in order:
    row += 1
    worksheet.write(row, col,     key)
    i =1
    for item in own[key]:
        worksheet.write(row, col + i, item)
        i += 1

workbook.close()


cursor.close()
connection.close()
FILE.close()

#data = cursor.fetchall()
#col_names = []
#for i in range(0, len(cursor.description)):
#    col_names.append(cursor.description[i][0])

##### Bag of Words model #######
cols_per_own = OrderedDict()

own.pop('SYS', None)
own.pop('EXFSYS', None)
own.pop('CTXSYS', None)

for k in own.keys():
    if 'MNH' not in k:
        own.pop(k, None)

for k in own.keys():
    av_counter = 0
    un_counter = 0
    lot = own[k]
    num = len(lot)
    col_names = []
    for t in lot:
        if t == 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA':
            pass
        else:
            query = 'SELECT column_name FROM ' + k + '.' + t
            print query
            cursor.execute(query)
            data = cursor.fetchall()
            for i in range(0, len(cursor.description)):
                col_names.append(cursor.description[i][0])
                av_counter += len(cursor.description)
            un_counter += len(list(set(col_names)))
    av_counter = float(av_counter/num)
    print 'number unique and average number of columns in {} : {} and {}'.format(k, un_counter, av_counter)
    cols_per_own[k] = (un_counter, av_counter, list(set(col_names)))

sql2 = 'select column_name from all_tab_columns'
cursor.execute(sql2)

for row in cursor:
    print row

sql3 = 'select column_name from all_tab_columns where owner = ' + k + 'and table_name = ' + t
cursor.execute(sql3)

sql4 = 'select column_name from user_tab_columns where table_name = MNH_BILLING.T_DOC_RG_BILLING' 
cursor.execute(sql4)

for row in cursor:
    print row
