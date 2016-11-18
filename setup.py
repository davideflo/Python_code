# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:03:43 2016

@author: d_floriello

setup file for compiler
"""

#from distutils.core import setup
#import py2exe
#import sys
#
#
#
#sys.setrecursionlimit(500000)
#
#setup(options = {"py2exe": {"dll_excludes": ["MSVCP90.dll"]}},console=['C:/Users/utente/Documents/Python Scripts/Disto2Excel_1.0.py'])


import sys
from cx_Freeze import setup, Executable

sys.setrecursionlimit(500000)

setup(
    name = "Convertitore fatture",
    version = "1.0",
    description = "Convertitore fatture da pdf a excel",
    executables = [Executable("C:/Users/utente/Documents/Python Scripts/Disto2Excel_1.py", base = "Win32GUI")])