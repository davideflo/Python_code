# Polynomial regression

import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 

data = pd.read_table("polregr.txt", header =None ,sep=r'\s+')

data_dict = {'X1': data[0], 'X2': data[1], 'price': data[2]}

data = pd.DataFrame(data = data_dict)

import matplotlib.pyplot as plt

plt.scatter(data['X1'], data['price']) #degree 2
plt.show()

plt.scatter(data['X2'], data['price']) #degree2
plt.show()

#interaction

plt.scatter(data['X1'], data['X2'])
plt.show()

model = Pipeline([('poly', PolynomialFeatures(degree = 2)), ('linear', LinearRegression(fit_intercept=True))])

model = model.fit(data[['X1', 'X2']], data['price'])

# coefficients of the model: [1, x1, x2, x1^2, x1x2, x2^2]
model.named_steps['linear'].coef_

new_data = pd.DataFrame({'X1': [0.05, 0.91, 0.31, 0.51], 'X2': [0.54, 0.91, 0.76, 0.31]})
expected_price = [180.38, 1312.07, 440.13, 343.72]

result = model.predict(new_data[['X1', 'X2']])

def norm_dist_exp(computed, expected):
	res = []
	for i in range(len(computed)):
		res.append(abs(computed[i] - expected[i])/expected[i])
	return(res)

res = norm_dist_exp(result, expected_price)

score = np.mean(res)