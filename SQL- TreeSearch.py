# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:33:59 2016

@author: utente

SQL - Tree Search
"""

from __future__ import division
from __future__ import unicode_literals
import cx_Oracle
import csv
from collections import OrderedDict
import sys
import time
import pandas as pd
import datetime


dns = cx_Oracle.makedsn('192.168.0.60', '1521', 'argon')
connection = cx_Oracle.connect('read_only', 'read_only', dns) 
cursor = connection.cursor()

sql_all = 'SELECT owner,table_name FROM dba_tables'

SQL = "select count(distinct ID_USER) from MNH_COMMON.R_CST_PROP_PRODOTTI"

#cursor.execute(SQL)
cursor.execute(sql_all)

own = OrderedDict()
macros = []

for row in cursor:
    print row
    macros.append(row[0])    

macro = list(set(macros))
### number of macro categories in the database -- owner
    
call = cursor.execute(sql_all)

for row in call:
    title = row[0]
    if title in own.keys():
        own[title].append(row[1])
    else:
        own[title] = ['AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA']
        own[title].append(row[1])        
        
        
for k in own.keys():
    if 'MNH' not in k:
        own.pop(k, None)



SQL =   """SELECT  v.id_contratto,
           v.cliente_codice,
           v.cliente_ragione_sociale,
           v.cliente_partita_iva,
           v.cliente_codice_fiscale,
           v.cliente_indirizzo,
           v.cliente_cap,
           v.cliente_comune,
           v.cliente_prov,
           v.cliente_gruppo,
           v.banca,
           v.cliente_consorzio,
           v.fornitura_codice,
           v.fornitura_indirizzo,
           v.fornitura_cap,
           v.fornitura_comune,
           v.fornitura_prov,
           v.fornitura_contratto_da,
           v.fornitura_contratto_a,
           v.fornitura_presa,
           v.fornitura_pod,
           v.fornitura_tipo_mis,
           v.fornitura_potenza,
           v.fornitura_tensione,
           v.trasporto_opz_tariff,
           v.trasporto_distributore,
           v.fattura_indirizzo,
           v.fattura_cap,
           v.fattura_comune,
           v.fattura_prov,
           v.banca_cliente,
           v.iban_cc,
           v.sottoscrittore_rid,
           v.sottoscrittore_codice_fiscale,
           v.perc_iva,
           v.art_esenzione_iva,
           v.doc_aut_iva,
           v.esenzione_imposta_erariale,
           v.esenzione_imposta_provinciale,
           v.frequenza_fatturaz,
           v.scadenza_pagamento,
           v.codice_rid,
           v.codice_rid_cliente,
           v.id_cliente,
           v.calcolo_perequazione,
           v.id_sito,
           (select s_label from mnh_common.t_cst_prop_enum forn_enum where id_enum= v.stato_fornitura and cd_cst_prop='STATO_FORNITURA') as stato_fornitura,
           v.note,
           v.d_valido_dal,
           v.d_valido_al,
           v.cd_tp_punto,
           v.agenzia,
           v.agente,
           v.descrizione_prodotto,
           v.prezzo_f0,
           v.prezzo_f1_fp,
           v.prezzo_f2_fop,
           v.prezzo_f3,
           v.consumo_contr_annuo,
           v.consumo_contr_f1_fp,
           v.consumo_contr_f2_fop,
           v.consumo_contr_f3,
           v.tipo_garanzia,
           v.referente,
           v.telefono_cliente,
           v.fax_cliente,
           v.mail_cliente,
           v.cod_cli_asso,
           v.versamento_diretto,
           v.codice_ditta,
           v.matricola_misuratore,
           v.matricola_correttore,
           v.coeff_c_misuratore,
           v.cod_remi,
           v.profilo_prelievo,
           v.accise_gas,
           v.cliente_gruppo_effettivo,
           v.stato_domiciliazione,
           v.tipo_di_rapporto,
           v.entita_fatturabile,
           v.cons_inizio_anno_solare,
           v.cons_inizio_anno_termico,
           v.SHIPPER,
           v.consumo_distributore,
           v.prezzo_pfor,
           v.prezzo_cmem,
           v.prezzo_pfisso, 
           v.prezzo_pfisso_axo, 
           v.codice_punto,
           v.codice_classe_misuratore,
           v.sconto_fedelta,
           v.recesso_sconto,
           v.potenza_impegnata,
           v.frequenza_lettura
           FROM (select * from mnh_common.v_aut_simil_template) as v
           WHERE  [Tipo punto:E:TP_PUNTO:14] = cd_tp_punto
           AND id_fornitore =  [#AZIENDA]
           """

SQL2 =  """SELECT  id_contratto,
           cliente_codice,
           cliente_ragione_sociale
           FROM mnh_common.v_aut_simil_template
           WHERE  [Tipo punto:E:TP_PUNTO:14] = cd_tp_punto
           AND id_fornitore =  [#AZIENDA]
           """

          

cursor.execute(SQL)
cursor.execute('select * from MNH_COMMON.V_AUT_SIMIL_TEMPLATE')

cursor.execute(SQL2)

