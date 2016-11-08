# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:35:02 2016

@author: d_floriello

Example Risk calculation
"""

import numpy as np
import sdeint
import matplotlib.pyplot as plt
from sympy.solvers.pde import pdsolve
from sympy import Function, diff, Eq, cos, sin
from sympy.abc import x, t


# Weiner Process Simulation
f = Function('f')
u = f(x, t)
ux = u.diff(x)
ut = u.diff(t)
uxx = ux.diff(x)
eq = Eq(ut + (1 - cos(2*x)*u + (x + 2 - sin(2*x))*ux - 0.5*(sin(x)**2)*uxx))
pdsolve(eq)


# Time steps
n = 1000 
T = 1  

###############################################################################
def Wiener(mu, sig, n, T):
    Delta = T/n
#    t = np.arange(1, step=Delta)
    S = np.zeros(n, np.dtype(float))
    x = range(1, len(S))
    for i in x:
        dW = np.random.normal(mu, sig)
        dt = np.sqrt(Delta)
        dS = dW*dt
        S[i] = S[i-1] + dS
    return S
###############################################################################

plt.figure()
plt.plot(Wiener(0,1,n,T))

def mu(x,t):
    return x**2

def SIG(x, t):
    return np.sqrt(x)

tspan = np.linspace(0.0, 1, 10001)
res = sdeint.itoint(mu, SIG, 1, tspan)

plt.figure()
plt.plot(tspan,res.ravel())

def mu2(x,t):
    return x + 2
def sig2(x,t):
    return np.sin(x)

plt.figure()
tspan2 = np.linspace(0.0, 1, 10001)
for i in range(10):
    plt.plot(tspan2, sdeint.itoint(mu2, sig2, 0, tspan2))
    
def ForwardKolmogorovEquation(drift, diffusion):
    p = Function('p')
    u = p(x, t)
    ut = u.diff(t)
    tdrift = u * drift
    tdiffusion = u * (1/2*diffusion**2)
    tdt = tdrift.diff(x)
    tdifft = tdiffusion.diff(x)
    tdifftt = tdifft.diff(x)
    eq = Eq(ut + tdt - tdifftt)
    return pdsolve(eq)
    
drift = Function('x + 2')
diffusion = Function('sin(x)')    

ForwardKolmogorovEquation(drift, diffusion)