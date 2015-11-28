# laptop battery 

import pandas as pd 
import numpy as np

data = pd.read_csv("laptopbattery.txt", header = None)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 

plt.scatter(data[0], data[1])
plt.show()

data_min = data.ix[data[0] <= 4]
data_over = data.ix[data[0] > 4]

f, axarr = plt.subplots(2, sharex=True)
axarr[0] = plt.scatter(data_min[0], data_min[1])
axarr[1] = plt.plot(data_over[0], data_over[1])
plt.show()


lm = LinearRegression(fit_intercept= True)

lmfit = lm.fit(data_min[0][:,np.newaxis], data_min[1])

lmpredict = lmfit.predict(data_min[0][:,np.newaxis])

plt.scatter(data_min[0], data_min[1])
plt.plot(data_min[0][:,np.newaxis], lmpredict, color='blue', linewidth=2)
plt.show()