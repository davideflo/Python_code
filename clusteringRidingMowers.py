import numpy as np 
import pandas as pd 
from sklearn import metrics
from sklearn.cluster import KMeans

data = pd.ExcelFile('RidingMowers.xls')

data = data.parse('Data')

nolabel = data[['Income', 'Lot_Size']]

import matplotlib.pyplot as plt 



plt.scatter(nolabel['Income'], nolabel['Lot_Size'])

plt.show()

km = KMeans(n_clusters=2, init='k-means++')
m2 = km.fit(nolabel)

estimators = {'kmeans2': KMeans(n_clusters=2, init='k-means++'), 'kmeans3':
	KMeans(n_clusters=3, init='k-means++'),
	'kmeans4': KMeans(n_clusters=4, init='k-means++')}

for k in range(2, 5):
	km = KMeans(n_clusters=k, init='k-means++')
	fit = km.fit(nolabel)
	labels = fit.labels_
	print(fit.inertia_)
	metrics.silhouette_score(nolabel, fit.labels_, metric='euclidean')

# best silhouette: k = 2. Ergo, il cinese ha fatto una pessima analisi.
