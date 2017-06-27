# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:18:09 2017

@author: utente

Queries per il report commerciale
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

txt = open('C:/Users/utente/Documents/SQL/Argon_File_1.txt','r').read()

sql = u''
with open('C:/Users/utente/Documents/SQL/Argon_File_1.txt','r') as inserts:
    for statement in inserts:
        sql += statement
#        cursor.execute(statement)

######## WORKS

sql_Argonfile2 = """SELECT id_contratto,
       cliente_codice,
       cliente_ragione_sociale,
       cliente_partita_iva,
       cliente_codice_fiscale,
       cliente_indirizzo,
       cliente_cap,
       cliente_comune,
       cliente_prov,
       cliente_gruppo,
       banca,
       cliente_consorzio,
       fornitura_codice,
       fornitura_indirizzo,
       fornitura_cap,
       fornitura_comune,
       fornitura_prov,
       fornitura_contratto_da,
       fornitura_contratto_a,
       fornitura_presa,
       fornitura_pod,
       fornitura_tipo_mis,
       fornitura_potenza,
       fornitura_tensione,
       trasporto_opz_tariff,
       trasporto_distributore,
       fattura_indirizzo,
       fattura_cap,
       fattura_comune,
       fattura_prov,
       banca_cliente,
       iban_cc,
       sottoscrittore_rid,
       sottoscrittore_codice_fiscale,
       perc_iva,
       art_esenzione_iva,
       doc_aut_iva,
       esenzione_imposta_erariale,
       esenzione_imposta_provinciale,
       frequenza_fatturaz,
       scadenza_pagamento,
       codice_rid,
       codice_rid_cliente,
       id_cliente,
       id_sito,
       d_valido_dal,
       d_valido_al,
       cd_tp_punto,
       agenzia,
       agente,
       descrizione_prodotto,
       prezzo_f0,
       prezzo_f1_fp,
       prezzo_f2_fop,
       prezzo_f3,
       consumo_contr_annuo,
       consumo_contr_f1_fp,
       consumo_contr_f2_fop,
       consumo_contr_f3,
       tipo_garanzia,
       referente,
       telefono_cliente,
       fax_cliente,
       mail_cliente,
       cod_cli_asso,
       versamento_diretto,
       codice_ditta,
       matricola_misuratore,
       matricola_correttore,
       coeff_c_misuratore,
       cod_remi,
       profilo_prelievo,
       accise_gas,
       cliente_gruppo_effettivo,
       stato_domiciliazione,
       tipo_di_rapporto,
       entita_fatturabile,
       cons_inizio_anno_solare,
       cons_inizio_anno_termico,
       SHIPPER,
       consumo_distributore,
       prezzo_pfor,
       prezzo_cmem,
       prezzo_pfisso, 
       prezzo_pfisso_axo, 
       codice_punto,
       codice_classe_misuratore,
       sconto_fedelta,
       recesso_sconto,
       calcolo_perequazione,
       (select s_label from mnh_common.t_cst_prop_enum forn_enum where id_enum= stato_fornitura and cd_cst_prop='STATO_FORNITURA') as stato_fornitura,
       note
       FROM mnh_common.v_aut_simil_template
       WHERE cd_tp_punto = 'EE' """


        
sql_Argonfile4 = """
select min (ag.cd_agenzia) cd_agenzia,
         min (az.s_denominazione) ds_agenzia,
         min (trim (ag.s_cognome || ' ' || ag.s_nome)) ds_agente,
         min (cl.s_denominazione) ds_cliente,
         fl.cd_punto,
         min (fl.s_pod_pdr) s_pod,
         min (ct.d_firma) d_firma,
         min (fl.d_iniz_forn) over (partition by fl.cd_contratto) d_contr_dal,
         case
            when sum (case when fl.d_fine_forn is null then 1 else 0 end) over (partition by fl.cd_contratto) = 0 then
               max (fl.d_fine_forn) over (partition by fl.cd_contratto)
            else
               null
         end
            d_contr_al,
         fl.cd_prodotto cd_listino,
         min (nvl (sc.n_cons_attualizz, sc.n_cons_contr)) n_energia_aa,
         max (case when extract (month from fp.d_periodo) = 1 then dm.n_valore else null end) n_cons_gen,
         max (case when extract (month from fp.d_periodo) = 2 then dm.n_valore else null end) n_cons_feb,
         max (case when extract (month from fp.d_periodo) = 3 then dm.n_valore else null end) n_cons_mar,
         max (case when extract (month from fp.d_periodo) = 4 then dm.n_valore else null end) n_cons_apr,
         max (case when extract (month from fp.d_periodo) = 5 then dm.n_valore else null end) n_cons_mag,
         max (case when extract (month from fp.d_periodo) = 6 then dm.n_valore else null end) n_cons_giu,
         max (case when extract (month from fp.d_periodo) = 7 then dm.n_valore else null end) n_cons_lug,
         max (case when extract (month from fp.d_periodo) = 8 then dm.n_valore else null end) n_cons_ago,
         max (case when extract (month from fp.d_periodo) = 9 then dm.n_valore else null end) n_cons_set,
         max (case when extract (month from fp.d_periodo) = 10 then dm.n_valore else null end) n_cons_ott,
         max (case when extract (month from fp.d_periodo) = 11 then dm.n_valore else null end) n_cons_nov,
         max (case when extract (month from fp.d_periodo) = 12 then dm.n_valore else null end) n_cons_dic,
         max (case when extract (month from stm.d_periodo) = 1 then stm.n_ener_attiva else null end) n_stima_gen,
         max (case when extract (month from stm.d_periodo) = 2 then stm.n_ener_attiva else null end) n_stima_feb,
         max (case when extract (month from stm.d_periodo) = 3 then stm.n_ener_attiva else null end) n_stima_mar,
         max (case when extract (month from stm.d_periodo) = 4 then stm.n_ener_attiva else null end) n_stima_apr,
         max (case when extract (month from stm.d_periodo) = 5 then stm.n_ener_attiva else null end) n_stima_mag,
         max (case when extract (month from stm.d_periodo) = 6 then stm.n_ener_attiva else null end) n_stima_giu,
         max (case when extract (month from stm.d_periodo) = 7 then stm.n_ener_attiva else null end) n_stima_lug,
         max (case when extract (month from stm.d_periodo) = 8 then stm.n_ener_attiva else null end) n_stima_ago,
         max (case when extract (month from stm.d_periodo) = 9 then stm.n_ener_attiva else null end) n_stima_set,
         max (case when extract (month from stm.d_periodo) = 10 then stm.n_ener_attiva else null end) n_stima_ott,
         max (case when extract (month from stm.d_periodo) = 11 then stm.n_ener_attiva else null end) n_stima_nov,
         max (case when extract (month from stm.d_periodo) = 12 then stm.n_ener_attiva else null end) n_stima_dic
    from mnh_common.v_punti_fornitura fl
         join mnh_common.t_prodotti pr
            on pr.cd_prodotto = fl.cd_prodotto
         left join mnh_common.t_agenti ag
            on ag.cd_agente = fl.cd_agente
         left join mnh_common.t_agenzie az
            on ag.cd_agenzia = az.cd_agenzia
         join mnh_common.t_clienti cl
            on cl.cd_cliente = fl.cd_intestatario
         left join mnh_common.t_contratti ct
            on ct.cd_contratto = fl.cd_contratto
         left join mnh_billing.t_stime_ee_aa sc
            on sc.cd_punto = fl.cd_punto
         left join mnh_billing.r_forniture_periodi fp
    on fp.id_fornitura = fl.id_fornitura and fp.d_periodo between to_date('01/01/2017','dd/mm/yyyy') and to_date('31/12/2017','dd/mm/yyyy')
         left join mnh_billing.t_doc_misure dm
            on     dm.id_documento = fp.id_doc_fatt
               and dm.d_periodo = fp.d_periodo
               and dm.cd_punto = fl.cd_punto
               and dm.cd_tp_grandezza_fatt = 'E_MM_TOT'
               and dm.cd_fascia = 'F0'
         left join mnh_logistica.t_misure_ee stm
            on     stm.cd_tp_misura = 'CURVE_STIMA'
               and stm.id_stato in (10, 20, 30)
               and stm.cd_punto = fl.cd_punto
               and stm.d_periodo = fp.d_periodo
   where fl.cd_tp_punto = 'EE'
       group by fl.cd_punto,
         fl.d_iniz_forn,
         fl.cd_contratto,
         fl.d_fine_forn,
         fl.cd_prodotto
"""

####################################################################################################

sql_Argonfile3 = """
SELECT riep.cd_agenzia,
         riep.ds_agenzia,
         dett.ds_agente,
         dett.ds_intestatario_fatt as ds_cliente,
         punti.cd_punto,
         dett.s_pod_pdr,
         contr.d_firma,
         punti.d_valido_dal,
         punti.d_valido_al,
         punti.cd_prodotto as cd_listino,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '1' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_gen,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '2' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_feb,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '3' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_mar,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '4' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_apr,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '5' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_mag,            
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '6' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_giu,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '7' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_lug,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '8' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_ago,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '9' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_set,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '10' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_ott,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '11' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_nov,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '12' AND dett.cd_regola_provv like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS gett_dic,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '1' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_gen,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '2' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_feb,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '3' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_mar,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '4' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_apr,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '5' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_mag,            
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '6' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_giu,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '7' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_lug,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '8' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_ago,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '9' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_set,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '10' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_ott,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '11' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_nov,
         SUM (CASE 
              WHEN EXTRACT (MONTH FROM riep.d_periodo) = '12' AND dett.cd_regola_provv not like '%GETTONE_%'
              THEN dett.n_imp
              ELSE 0
              END) AS provv_dic
    FROM mnh_crm.v_age_calcolo_provvigioni riep
         JOIN mnh_crm.v_age_remunerazioni_dett dett
            ON dett.id_remunerazione = riep.id_remunerazione
         LEFT JOIN mnh_common.v_forniture_punti_lookup punti
            ON punti.s_pod_pdr = dett.s_pod_pdr
         LEFT JOIN mnh_common.t_contratti contr
            ON contr.cd_contratto = punti.cd_contratto
   WHERE TO_CHAR (riep.d_periodo, 'yyyy') = '2017'
       and riep.id_fornitore =  1
GROUP BY riep.cd_agenzia,
         riep.ds_agenzia,
         dett.ds_agente,
         dett.ds_intestatario_fatt,
         punti.cd_punto,
         dett.s_pod_pdr,
         contr.d_firma,
         punti.d_valido_dal,
         punti.d_valido_al,
         punti.cd_prodotto
"""
####################################################################################################

sql= """
with ta as (
-- T8 marg
select /*+ materialize */ * from (
SELECT sub.id_fornitura,
       to_char(extract( month from (trunc (sub.d_periodo,'mm')))) d_periodo,
SUM(
       CASE WHEN vend_ee_f1_prz IS NOT NULL THEN NVL(ROUND (vend_ee_f1_qta * mnh_report.get_prz_acq 
        (rdp.cd_prodotto,'ACQ_F1',to_date('01/'||extract( month from rdp.d_periodo) ||'/'||extract( year from rdp.d_periodo) ,'dd/mm/yyyy')),2),0) ELSE 0 END + -- IMP_ACQ_F1,
       CASE WHEN vend_ee_f2_prz IS NOT NULL THEN NVL(ROUND 
       (vend_ee_f2_qta * mnh_report.get_prz_acq 
        (rdp.cd_prodotto,'ACQ_F2',to_date('01/'||extract( month from rdp.d_periodo) ||'/'||extract( year from rdp.d_periodo) ,'dd/mm/yyyy')),2),0) ELSE 0 END + -- IMP_ACQ_F2,
       CASE WHEN vend_ee_f3_prz IS NOT NULL THEN NVL(ROUND (vend_ee_f3_qta * mnh_report.get_prz_acq
        (rdp.cd_prodotto,'ACQ_F3',to_date('01/'||extract( month from rdp.d_periodo) ||'/'||extract( year from rdp.d_periodo) ,'dd/mm/yyyy')),2),0) ELSE 0 END + -- IMP_ACQ_F3,
       CASE WHEN vend_ee_f0_prz IS NOT NULL THEN NVL(ROUND (vend_ee_f0_qta * mnh_report.get_prz_acq 
        (rdp.cd_prodotto,'ACQ_F0',to_date('01/'||extract( month from rdp.d_periodo) ||'/'||extract( year from rdp.d_periodo) ,'dd/mm/yyyy')),2),0) ELSE 0 END + -- IMP_ACQ_F0,
       CASE WHEN vend_ee_fp_prz IS NOT NULL THEN NVL(ROUND (vend_ee_fp_qta * mnh_report.get_prz_acq 
            (rdp.cd_prodotto,'ACQ_F1',to_date('01/'||extract( month from rdp.d_periodo) ||'/'||extract( year from rdp.d_periodo) ,'dd/mm/yyyy')),2),0) ELSE 0 END + -- IMP_ACQ_FP,
       CASE WHEN vend_ee_fop_prz IS NOT NULL THEN NVL(ROUND (vend_ee_fop_qta * mnh_report.get_prz_acq 
             (rdp.cd_prodotto,'ACQ_F2',to_date('01/'||extract( month from rdp.d_periodo) ||'/'||extract( year from rdp.d_periodo) ,'dd/mm/yyyy')),2),0) ELSE 0 END + -- IMP_ACQ_FOP
       CASE WHEN pun_aritm_f1_prz IS NOT NULL THEN  NVL(ROUND (pun_aritm_f1_qta * (pun_aritm_f1_prz - NVL((COALESCE(cp_punti_pun.n_valore, cp_prod_pun.n_valore)), 0)/1000  + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)),2), 0)  ELSE 0  END +
       --NVL(pun_aritm_f1_imp,0) + --,
       CASE WHEN pun_f1_prz IS NOT NULL THEN  NVL(ROUND (pun_f1_qta * (pun_f1_prz + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)), 2), 0)  ELSE 0  END +
       --NVL(pun_f1_imp,0) + 
       CASE WHEN gppb_f1_prz IS NOT NULL THEN  NVL(ROUND (gppb_f1_qta * (gppb_f1_prz + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)), 2), 0)  ELSE 0  END +
       --NVL(gppb_f1_imp,0) + --,
       CASE WHEN pun_aritm_f2_prz IS NOT NULL THEN  NVL(ROUND (pun_aritm_f2_qta * (pun_aritm_f2_prz - NVL((COALESCE(cp_punti_pun.n_valore, cp_prod_pun.n_valore)), 0)/1000 + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)), 2), 0)  ELSE 0  END +
       --NVL(pun_aritm_f2_imp,0) + --,
       CASE WHEN pun_f2_prz IS NOT NULL THEN  NVL(ROUND (pun_f2_qta * (pun_f2_prz + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)), 2), 0)  ELSE 0  END +
       --NVL(pun_f2_imp,0) + --,
       CASE WHEN gppb_f2_prz IS NOT NULL THEN  NVL(ROUND (gppb_f2_qta * (gppb_f2_prz + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)), 2), 0)  ELSE 0  END +
      -- NVL(gppb_f2_imp,0) + --, 
       CASE WHEN pun_aritm_f3_prz IS NOT NULL THEN  NVL(ROUND (pun_aritm_f3_qta * (pun_aritm_f3_prz - NVL((COALESCE(cp_punti_pun.n_valore, cp_prod_pun.n_valore)), 0)/1000 + NVL((COALESCE(cp_punti.n_valore, cp_prod.n_valore)), 0)), 2), 0)  ELSE 0  END +
       --NVL(pun_aritm_f3_imp,0) + --,
       CASE WHEN pun_f3_prz IS NOT NULL THEN  NVL(ROUND (pun_f3_qta * (pun_f3_prz + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0)), 2), 0)  ELSE 0  END +
      -- NVL(pun_f3_imp,0) + --,
       CASE WHEN gppb_f3_prz IS NOT NULL THEN  NVL(ROUND (gppb_f3_qta * gppb_f3_prz + NVL(COALESCE(cp_punti.n_valore, cp_prod.n_valore), 0), 2), 0)  ELSE 0  END 
      -- NVL(gppb_f3_imp,0)
       ) as somma_mese      
  FROM mnh_billing.t_documenti doc 
     join  mnh_billing.v_report_componenti_margini SUB on (doc.id_documento=sub.id_documento and doc.id_stato in (40,60)
                 )
      join mnh_billing.r_doc_periodi rdp 
       on (sub.id_documento=rdp.id_documento and sub.d_periodo=rdp.d_periodo and sub.id_fornitura=rdp.id_fornitura
       and TRUNC(rdp.d_periodo,'YYYY') = to_date('01/01/2017','dd/mm/yyyy')
       )
        LEFT  JOIN mnh_common.V_SYS_CST_PROP_PUNTI cp_punti
                         on cp_punti.cd_punto = rdp.cd_punto
                        and cp_punti.CD_CST_PROP = 'ACQ_SPREAD_ENERGIA'
                        and   rdp.d_periodo between cp_punti.D_VALIDO_DAL and nvl(cp_punti.D_VALIDO_AL, rdp.d_periodo)
        LEFT  JOIN mnh_common.V_SYS_CST_PROP_PRODOTTI cp_prod
                        ON cp_prod.cd_prodotto = rdp.cd_prodotto
                        and   cp_prod.CD_CST_PROP = 'ACQ_SPREAD_ENERGIA'
                        and   rdp.d_periodo between cp_prod.D_VALIDO_DAL and nvl(cp_prod.D_VALIDO_AL, rdp.d_periodo)
        LEFT JOIN mnh_common.V_SYS_CST_PROP_PUNTI cp_punti_pun
                         on cp_punti_pun.cd_punto = rdp.cd_punto
                        and cp_punti_pun.CD_CST_PROP = 'SPREAD_PUN'
                        and   rdp.d_periodo between cp_punti_pun.D_VALIDO_DAL and nvl(cp_punti_pun.D_VALIDO_AL, rdp.d_periodo)
        LEFT JOIN mnh_common.V_SYS_CST_PROP_PRODOTTI cp_prod_pun
                        ON cp_prod_pun.cd_prodotto = rdp.cd_prodotto
                        and   cp_prod_pun.CD_CST_PROP = 'SPREAD_PUN'
                        and   rdp.d_periodo between cp_prod_pun.D_VALIDO_DAL and nvl(cp_prod_pun.D_VALIDO_AL, rdp.d_periodo)                                              
  group by sub.id_fornitura,trunc (sub.d_periodo,'mm')) tt
pivot (
max(somma_mese)
FOR d_periodo 
   in ('1' as gen, '2' as feb,  '3' as mar, '4' as apr, '5' as mag, '6' as giu, '7' as lug,
        '8' as ago, '9' as sett, '10' as ott, '11' as nov, '12' as dic) )) 
-- Inizio query      
select * from (
select  az.s_denominazione AS agenzia,
        ag.s_nome || ' ' || ag.s_cognome AS agente,
        temp.s_pod AS pod,
        MAX (pot.ragione_sociale)
              KEEP (DENSE_RANK LAST ORDER BY doc.id_documento)
              AS ragione_sociale,
           MAX (pot.s_indirizzo_forn)  KEEP (DENSE_RANK LAST ORDER BY doc.id_documento) AS indirizzo,
           MAX (pot.ds_localita_forn) KEEP (DENSE_RANK LAST ORDER BY doc.id_documento) AS citta,
        PRODOTTO_1,
        PRODOTTO_1_DA,
        PRODOTTO_2,
        PRODOTTO_2_DA,
        PRODOTTO_3,
        PRODOTTO_3_DA,
        PRODOTTO_4,
        PRODOTTO_4_DA,
        PRODOTTO_5,
        PRODOTTO_5_DA,
        PRODOTTO_6,
        PRODOTTO_6_DA,
        MIN (inizio_forniura_pod) AS inizio_fornitura_pod,
        MAX (fine_fornitura_pod) AS fine_fornitura_pod,
        CASE
              WHEN SYSDATE BETWEEN MIN (pot.inizio_forniura_pod)
                               AND  MAX (fine_fornitura_pod)
                   OR (SYSDATE >= MIN (pot.inizio_forniura_pod)
                       AND MAX (fine_fornitura_pod) IS NULL)
              THEN
                 'IN FORNITURA'
              WHEN SYSDATE < MIN (pot.inizio_forniura_pod)
              THEN
                 'FUTURA FORNITURA'
              ELSE
                 'USCITO'
           END
              AS stato_attuale_POD,
       MAX (NVL (pot.n_potenza_imp, pot.n_potenza_disp)) AS potenza_kW,
      SUM(CASE
          WHEN rb.cd_componente LIKE '%PCV%' THEN rb.n_imp
          ELSE NULL
       END) pcv,
       COUNT (DISTINCT TRUNC (rdp.d_periodo, 'mm')) AS num_mesi_fatturati,
        sum (case when extract (month from rdp.d_periodo) = 1 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
           max(marg.gen)*(max(parv.n_valore / 100) + 1) margine_gen,
       sum (case when extract (month from rdp.d_periodo) = 2 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
          max(marg.feb)*(max(parv.n_valore / 100) + 1) margine_feb,
       sum (case when extract (month from rdp.d_periodo) = 3 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
          max(marg.mar)*(max(parv.n_valore / 100) + 1) margine_mar,
       sum (case when extract (month from rdp.d_periodo) = 4 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
            max(marg.apr)*(max(parv.n_valore / 100) + 1) margine_apr,
       sum (case when extract (month from rdp.d_periodo) = 5 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
           max(marg.mag)*(max(parv.n_valore / 100) + 1) margine_mag,
       sum (case when extract (month from rdp.d_periodo) = 6 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
            max(marg.giu)*(max(parv.n_valore / 100) + 1) margine_giu,
       sum (case when extract (month from rdp.d_periodo) = 7 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
             max(marg.lug)*(max(parv.n_valore / 100) + 1) margine_lug,
       sum (case when extract (month from rdp.d_periodo) = 8 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
         max(marg.ago)*(max(parv.n_valore / 100) + 1) margine_ago,
       sum (case when extract (month from rdp.d_periodo) = 9 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
          max(marg.sett)*(max(parv.n_valore / 100) + 1) margine_sett,
       sum (case when extract (month from rdp.d_periodo) = 10 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
         max(marg.ott)*(max(parv.n_valore / 100) + 1) margine_ott,
       sum (case when extract (month from rdp.d_periodo) = 11 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
          max(marg.nov)*(max(parv.n_valore / 100) + 1) margine_nov,
       sum (case when extract (month from rdp.d_periodo) = 12 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') and c.cd_componente not in ('ONERE_GARANZIA_EE','ONERE_SPOT','POWER_ZOOM','ONERE_SPOT_EE') then rb.n_imp else null end) -
          max(marg.dic)*(max(parv.n_valore / 100) + 1) margine_dic,
       sum (case when extract (month from rdp.d_periodo) = 1 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE')  then  rb.n_imp else null end) fatt_energia_gen,
       sum (case when extract (month from rdp.d_periodo) = 2 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE')   then  rb.n_imp else null end) fatt_energia_feb,
       sum (case when extract (month from rdp.d_periodo) = 3 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') then  rb.n_imp else null end) fatt_energia_mar,
       sum (case when extract (month from rdp.d_periodo) = 4 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE')  then  rb.n_imp else null end) fatt_energia_apr,
       sum (case when extract (month from rdp.d_periodo) = 5 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE')  then  rb.n_imp else null end) fatt_energia_mag,
       sum (case when extract (month from rdp.d_periodo) = 6 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE')  then  rb.n_imp else null end) fatt_energia_giu,
       sum (case when extract (month from rdp.d_periodo) = 7 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE')  then  rb.n_imp else null end) fatt_energia_lug,
       sum (case when extract (month from rdp.d_periodo) = 8 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') then  rb.n_imp else null end) fatt_energia_ago,
       sum (case when extract (month from rdp.d_periodo) = 9 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') then  rb.n_imp else null end) fatt_energia_set,
       sum (case when extract (month from rdp.d_periodo) = 10 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') then  rb.n_imp else null end) fatt_energia_ott,
       sum (case when extract (month from rdp.d_periodo) = 11 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') then  rb.n_imp else null end) fatt_energia_nov,
       sum (case when extract (month from rdp.d_periodo) = 12 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE','SCONTO_EE','VARIE_EE') then  rb.n_imp else null end) fatt_energia_dic,
       sum (case when extract (month from rdp.d_periodo) = 1 then  rb.n_imp else null end) fatturato_tot_forn_gen,
       sum (case when extract (month from rdp.d_periodo) = 2 then  rb.n_imp else null end) fatturato_tot_forn_feb,
       sum (case when extract (month from rdp.d_periodo) = 3 then  rb.n_imp else null end) fatturato_tot_forn_mar,
       sum (case when extract (month from rdp.d_periodo) = 4 then  rb.n_imp else null end) fatturato_tot_forn_apr,
       sum (case when extract (month from rdp.d_periodo) = 5 then  rb.n_imp else null end) fatturato_tot_forn_mag,
       sum (case when extract (month from rdp.d_periodo) = 6 then  rb.n_imp else null end) fatturato_tot_forn_giu,
       sum (case when extract (month from rdp.d_periodo) = 7 then  rb.n_imp else null end) fatturato_tot_forn_lug,
       sum (case when extract (month from rdp.d_periodo) = 8 then  rb.n_imp else null end) fatturato_tot_forn_ago,
       sum (case when extract (month from rdp.d_periodo) = 9 then  rb.n_imp else null end) fatturato_tot_forn_set,
       sum (case when extract (month from rdp.d_periodo) = 10 then  rb.n_imp else null end) fatturato_tot_forn_ott,
       sum (case when extract (month from rdp.d_periodo) = 11 then  rb.n_imp else null end) fatturato_tot_forn_nov,
       sum (case when extract (month from rdp.d_periodo) = 12 then  rb.n_imp else null end) fatturato_tot_forn_dic,
       sum (case when extract (month from rdp.d_periodo) = 1 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_gen,
       sum (case when extract (month from rdp.d_periodo) = 2 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_feb,
       sum (case when extract (month from rdp.d_periodo) = 3 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_mar,
       sum (case when extract (month from rdp.d_periodo) = 4 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_apr,
       sum (case when extract (month from rdp.d_periodo) = 5 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_mag,
       sum (case when extract (month from rdp.d_periodo) = 6 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_giu,
       sum (case when extract (month from rdp.d_periodo) = 7 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_lug,
       sum (case when extract (month from rdp.d_periodo) = 8 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_ago,
       sum (case when extract (month from rdp.d_periodo) = 9 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_set,
       sum (case when extract (month from rdp.d_periodo) = 10 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_ott,
       sum (case when extract (month from rdp.d_periodo) = 11 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_nov,
       sum (case when extract (month from rdp.d_periodo) = 12 and c.cd_tp_riga in ('PREZZO_EE', 'PERDITE_EE')and c.cd_componente not in('PREZZO_UNICO_CONG_F0','INDICE_PUN') then  rb.n_qta else null end) consumo_dic,
       MAX (gett_age_pod.gett_agente) AS gettone_agente,
       MAX (fee_gen) fee_gen,
       MAX (fee_feb) fee_feb,
       MAX (fee_mar) fee_mar,
       MAX (fee_apr) fee_apr,
       MAX (fee_mag) fee_mag,
       MAX (fee_giu) fee_giu,
       MAX (fee_lug) fee_lug,
       MAX (fee_ago) fee_ago,
       MAX (fee_set) fee_set,
       MAX (fee_ott) fee_ott,
       MAX (fee_nov) fee_nov,
       MAX (fee_dic) fee_dic,
           MAX (CASE
                WHEN PRODOTTO_6 is not null THEN (select MAX(N_VALORE  * 1000) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_6 and  cd_cst_prop = 'CORR_SQUIL_PEREQ')
                WHEN PRODOTTO_5 is not null THEN (select MAX(N_VALORE  * 1000) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_5 and cd_cst_prop = 'CORR_SQUIL_PEREQ')  
                WHEN PRODOTTO_4 is not null THEN (select MAX(N_VALORE  * 1000) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_4 and cd_cst_prop = 'CORR_SQUIL_PEREQ')  
                WHEN PRODOTTO_3 is not null THEN (select MAX(N_VALORE  * 1000) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_3 and cd_cst_prop = 'CORR_SQUIL_PEREQ')  
                WHEN PRODOTTO_2 is not null THEN (select MAX(N_VALORE  * 1000) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_2 and cd_cst_prop = 'CORR_SQUIL_PEREQ')     
                WHEN PRODOTTO_1 is not null THEN (select MAX(N_VALORE  * 1000) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_1 and cd_cst_prop = 'CORR_SQUIL_PEREQ')                                                                                                                                                                                                      
                END)
                PRZ_PEREQ,
     SUM(CASE
      WHEN rb.cd_componente = 'CORR_SQUIL_PEREQ' THEN rb.n_imp
      ELSE NULL
   END)
  importo_pereq,
             MAX (CASE
                WHEN PRODOTTO_6 is not null THEN (select MAX(N_VALORE) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_6 and  cd_cst_prop = 'SCONTO_FEDELTA')
                WHEN PRODOTTO_5 is not null THEN (select MAX(N_VALORE) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_5 and cd_cst_prop = 'SCONTO_FEDELTA')  
                WHEN PRODOTTO_4 is not null THEN (select MAX(N_VALORE) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_4 and cd_cst_prop = 'SCONTO_FEDELTA')  
                WHEN PRODOTTO_3 is not null THEN (select MAX(N_VALORE) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_3 and cd_cst_prop = 'SCONTO_FEDELTA')  
                WHEN PRODOTTO_2 is not null THEN (select MAX(N_VALORE) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_2 and cd_cst_prop = 'SCONTO_FEDELTA')     
                WHEN PRODOTTO_1 is not null THEN (select MAX(N_VALORE) from mnh_billing.r_cst_prop_prodotti 
                                                  where cd_prodotto = PRODOTTO_1 and cd_cst_prop = 'SCONTO_FEDELTA')                                                                                                                                                                                                      
                END)
                SCONTO_FEDELTA,
                MAX(cp_recesso.n_valore) RECESSO_SCONTO
FROM
--- T1 temp
 (
      SELECT 
          s_pod,
          id_fornitura,
          MAX(DECODE (q.occorrenza, 1, q.cd_prodotto)) AS PRODOTTO_1,
          MAX(DECODE (q.occorrenza, 1, q.da)) AS PRODOTTO_1_DA,
          MAX(DECODE (q.occorrenza, 2, q.cd_prodotto)) AS PRODOTTO_2,
          MAX(DECODE (q.occorrenza, 2, q.da)) AS PRODOTTO_2_DA,
          MAX(DECODE (q.occorrenza, 3, q.cd_prodotto)) AS PRODOTTO_3,
          MAX(DECODE (q.occorrenza, 3, q.da)) AS PRODOTTO_3_DA,
          MAX(DECODE (q.occorrenza, 4, q.cd_prodotto)) AS PRODOTTO_4,
          MAX(DECODE (q.occorrenza, 4, q.da)) AS PRODOTTO_4_DA,
          MAX(DECODE (q.occorrenza, 5, q.cd_prodotto)) AS PRODOTTO_5,
          MAX(DECODE (q.occorrenza, 5, q.da)) AS PRODOTTO_5_DA,
          MAX(DECODE (q.occorrenza, 6, q.cd_prodotto)) AS PRODOTTO_6,
          MAX(DECODE (q.occorrenza, 6, q.da)) AS PRODOTTO_6_DA
   FROM   
    (SELECT y.occorrenza, x.s_pod, x.da,  x.cd_prodotto, x.id_fornitura
     FROM   (SELECT a.*, row_number () OVER (PARTITION BY a.s_pod,a.id_fornitura  ORDER BY da) AS riga
               FROM (SELECT s_pod, vfl.cd_prodotto, MIN(vfl.d_valido_dal) AS da,  id_fornitura
                       FROM mnh_common.v_forniture_lookup vfl, mnh_common.t_punti_el_st tpes
                      WHERE vfl.cd_punto = tpes.cd_punto
                      GROUP BY s_pod, vfl.cd_prodotto, vfl.id_fornitura) a
            ) x,
            (SELECT LEVEL occorrenza FROM DUAL CONNECT BY LEVEL <= 6) y
    WHERE y.occorrenza = x.riga(+)
   ) q
   WHERE   s_pod IS NOT NULL 
GROUP BY   s_pod,id_fornitura) temp
--T9
    JOIN
      (  SELECT   vfpl.s_pod_pdr,
                  vfpl.cd_agente,
                  vfpl.id_fornitura,--agg
                  MAX (vfpl.n_potenza_imp)
                     KEEP (DENSE_RANK LAST ORDER BY vfpl.d_valido_dal)
                     AS n_potenza_imp,
                  MAX (vfpl.n_potenza_disp)
                     KEEP (DENSE_RANK LAST ORDER BY vfpl.d_valido_dal)
                     AS n_potenza_disp,
                  MIN (vfpl.d_iniz_forn) AS inizio_forniura_pod,
                  MAX (vfpl.d_fine_forn)
                     KEEP (DENSE_RANK LAST ORDER BY vfpl.d_valido_dal)
                     AS fine_fornitura_pod,
                     max(tef.s_destinatario)    KEEP (DENSE_RANK LAST ORDER BY vfpl.d_valido_dal) as ragione_sociale,
                     max(vfpl.s_indirizzo_forn) KEEP (DENSE_RANK LAST ORDER BY vfpl.d_valido_dal) as s_indirizzo_forn,
                     max(vfpl.ds_localita_forn) KEEP (DENSE_RANK LAST ORDER BY vfpl.d_valido_dal) as ds_localita_forn
           FROM   mnh_common.v_forniture_punti_lookup vfpl, mnh_common.t_entita_fatt tef
          WHERE   cd_tp_punto = 'EE'
          and vfpl.cd_entita_fatt = tef.cd_entita_fatt
       GROUP BY   s_pod_pdr,
                  cd_agente,--aggiunto
                  id_fornitura--aggiunto
       ) pot
   ON pot.s_pod_pdr = temp.s_pod and pot.id_fornitura=temp.id_fornitura
join ta marg on marg.id_fornitura= temp.id_fornitura  
--T2 rdp
   LEFT OUTER JOIN mnh_billing.r_doc_periodi rdp
    ON rdp.s_pod_pdr = temp.s_pod --
       and rdp.id_fornitura=temp.id_fornitura
      AND TRUNC (rdp.d_periodo, 'yyyy') = to_date('01/01/2017','dd/mm/yyyy')
      and rdp.id_documento in ( select id_documento from  mnh_billing.t_documenti doc
                    where doc.id_stato IN (40, 60)  
                            and cd_sezionale='FATT_EE')                           
--T3 doc
   LEFT OUTER JOIN mnh_billing.t_documenti doc
       on   doc.id_documento = rdp.id_documento
--T4 rb
    left join mnh_billing.t_doc_rg_billing rb
     on rb.id_documento = rdp.id_documento
       and rb.id_fornitura = rdp.id_fornitura
       and rb.d_periodo = rdp.d_periodo
       and rb.fl_contabilizza = 'S'
       and rb.cd_componente not like 'IVA_%'
--T5 c     
    LEFT OUTER JOIN mnh_billing.t_componenti c
             ON c.cd_componente = rb.cd_componente                          
--T6 par           
    LEFT OUTER join 
(select par.cd_tp_tensione, parv.* from
mnh_billing.t_parametri par
   join mnh_billing.t_param_valori parv
     on parv.id_parametro = par.id_parametro
     and par.cd_parametro = 'PERDITE_EE') parv 
 on rdp.cd_tp_tensione = parv.cd_tp_tensione    
    and rdp.d_periodo between parv.d_valido_dal and nvl ( parv.d_valido_al, rdp.d_periodo)
--T10  ag     
    LEFT OUTER JOIN mnh_common.t_agenti ag ON ag.cd_agente = pot.cd_agente
--T11  az  
     JOIN mnh_common.t_agenzie az ON ag.cd_agenzia = az.cd_agenzia
--T12  gett_age_pod  
      LEFT OUTER JOIN
                   (  SELECT   gett.cd_agente,
                               to_date('01/01/2017','dd/mm/yyyy') 
                               AS perido_anno,
                                od.s_pod,
                                SUM (gett.n_imp) AS gett_agente
                         FROM   mnh_crm.t_age_remunerazioni prov,
                                mnh_crm.t_age_remunerazioni_dett gett,
                                mnh_crm.t_ord_dett od
                        WHERE   prov.id_stato in (20, 30)
                                AND gett.id_remunerazione = prov.id_remunerazione
                                AND gett.cd_regola_provv IN
                                         ('GETT_FIGLIA',
                                          'GETTONE_CONF_EE',
                                          'GETTONE_EE',
                                          'GETTONE_EE_CONTR',
                                          'GETTONE_13M_EE')
                                AND gett.id_ordine = od.id_ordine
                                AND gett.n_riga = od.n_riga
                                and TRUNC (gett.d_periodo, 'yyyy') = to_date('01/01/2017','dd/mm/yyyy')
                     GROUP BY   gett.cd_agente,
                                od.s_pod) gett_age_pod
                 ON gett_age_pod.cd_agente = ag.cd_agente
                    AND gett_age_pod.perido_anno =
                          TRUNC (rdp.d_periodo, 'yyyy')
                    AND gett_age_pod.s_pod = rdp.s_pod_pdr
--T13                        
      LEFT OUTER JOIN
                 (select * from mnh_crm.v_report_fee) fee_age
              ON     fee_age.cd_agente = doc.cd_agente
                 AND fee_age.id_documento = doc.id_documento
                 AND fee_age.id_fornitura = rdp.id_fornitura  
                 AND fee_age.d_periodo = rdp.d_periodo
--
       LEFT OUTER JOIN
      mnh_common.V_SYS_CST_PROP_PUNTI cp_recesso
                         on cp_recesso.cd_punto = rdp.cd_punto
                        and cp_recesso.CD_CST_PROP = 'RECESSO_SCONTO'
                        and   rdp.d_periodo between cp_recesso.D_VALIDO_DAL and nvl(cp_recesso.D_VALIDO_AL, rdp.d_periodo)                                            
group by  temp.s_pod,
           ag.cd_agenzia,
           ag.cd_agente,
           az.s_denominazione,
           ag.s_nome || ' ' || ag.s_cognome,
           PRODOTTO_1,
           PRODOTTO_1_DA,
           PRODOTTO_2,
           PRODOTTO_2_DA,
           PRODOTTO_3,
           PRODOTTO_3_DA,
           PRODOTTO_4,
           PRODOTTO_4_DA,
           PRODOTTO_5,
           PRODOTTO_5_DA,
           PRODOTTO_6,
           PRODOTTO_6_DA)"""

sql = "select * from MNH_REPORT.V__RPT__000000000217"

cursor.execute(sql)
