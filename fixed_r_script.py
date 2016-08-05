# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 10:54:51 2016

@author: utente

Calling R script to generate the fixed datasets for h2o modelling
"""
### jar file for python: C:\Users\utente\Anaconda2\h2o_jar
"""
 Examples
  --------
  Using the 'proxy' parameter

  >>> import h2o
  >>> import urllib
  >>> proxy_dict = urllib.getproxies()
  >>> h2o.init(proxy=proxy_dict)
  Starting H2O JVM and connecting: ............... Connection successful!

"""
import h2o
from pyper import *

h2o.init()

r = R("C://Program Files//R//R-3.3.1//bin//R")

r.run("source('C://Users//utente//Documents//R_code//functions_for_PUN_server.R')")
r.run("source('C://Users//utente//Documents//R_code//functions_for_PPIA_server.R')")
r.run("library(h2o)")
r.run("h2o.init(nthreads = -1, max_mem_size = '20g')")
r.run("source('C://Users//utente//Documents//R_code//R_to_P.R')")

