#### Theano Tutorial

import numpy as np 
import theano
import theano.tensor as T 

# Exercise baby steps

a = T.vector()
out = a + a ** 10
f = theano.function([a], out)

