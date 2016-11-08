# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:22:57 2016

@author: utente

sql connections
"""
####################################################################################################
#import os
from __future__ import division
from __future__ import unicode_literals
import cx_Oracle
import csv
from collections import OrderedDict
import sys
import time

reload(sys)  
sys.setdefaultencoding('utf8')
 
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
####################################################################################################
def dict_writer(name, diz):
    import xlsxwriter    
    workbook = xlsxwriter.Workbook(name + '.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    
    order=diz.keys()
    for key in order:
        row += 1
        worksheet.write(row, col,     key)
        i =1
        for item in diz[key]:
            worksheet.write(row, col + i, item)
            i += 1
    
    workbook.close()
####################################################################################################

#data = cursor.fetchall()
#col_names = []
#for i in range(0, len(cursor.description)):
#    col_names.append(cursor.description[i][0])

##### Bag of Words model #######

own.pop('SYS', None)
own.pop('EXFSYS', None)
own.pop('CTXSYS', None)

for k in own.keys():
    if 'MNH' not in k:
        own.pop(k, None)

cols_per_own = OrderedDict()
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
            query = 'SELECT * FROM ' + k + '.' + t + ' where 1=0'
            print query
            try:
                cursor.execute(query)
                cur_descr = cursor.description
            except:
                #print 'Error, but column names:'                
                pass            
            print cursor.description
            cur_descr = cursor.description                
            #data = cursor.fetchall()
            for i in range(0, len(cur_descr)):
                col_names.append(cur_descr[i][0])
                av_counter += len(cur_descr)
            un_counter += len(list(set(col_names)))
    av_counter = float(av_counter/num)
    print 'number unique and average number of columns in {} : {} and {}'.format(k, un_counter, av_counter)
    cpo = list(set(col_names))
    cpo.insert(0, av_counter)
    cpo.insert(0, un_counter)
    cols_per_own[k] = cpo

dict_writer('colonne', cols_per_own)

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

sql5 = 'select * from MNH_BILLING.T_MIS_EE_HH where 1=0' 
try:
    cursor.execute(sql5)
except:
    pass
cd2 = cursor.description
print cd2

mb = cols_per_own['MNH_BILLING']
'OBJ_SERIE' in mb[2]

SQL = 'SELECT * FROM MNH_LOGISTICA.T_MISURE_EE_FF'
cursor.execute(SQL)
data = cursor.fetchall()

#### cerca 'F009996-Canale indiretto' nel database ##########
start = time.time()
lf = 'F009996-Canale indiretto'
list_of_tables = []
list_of_cols = []
for k in own.keys():
    lot = own[k]
    for t in lot:
        if t == 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA':
            pass
        else:
            query = 'SELECT * FROM ' + k + '.' + t + ' where 1=0'
            print query
            try:
                cursor.execute(query)
                cur_descr = cursor.description
            except:
                #print 'Error, but column names:'                
                pass            
            print cursor.description
            cur_descr = cursor.description                
            #data = cursor.fetchall()
            for i in range(0, len(cur_descr)):
                col_type = str(cur_descr[i][1])
                if col_type == "<type 'cx_Oracle.STRING'>" or col_type == "<type 'cx_Oracle.CHAR'>":
                    col_name = cur_descr[i][0]
                    sql_loc = 'select ' + col_name + ' from ' + k + '.' + t
                    cursor.execute(sql_loc)
                    data = cursor.fetchall()
                    ldata = [d[0] for d in data]
                    if lf in ldata:
                        list_of_tables.append(k + '.' + t)
                        list_of_cols.append(col_name)
end = time.time()
print end - start                       
                        
                        
                        
                        
                        
                        
cursor.execute('select S_DENOMINAZIONE from MNH_COMMON.T_AGENZIE')                       
data = cursor.fetchall()
ldata = [d[0] for d in data]
lf in ldata

