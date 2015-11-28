# Blood Driven Data

import pandas as pd 
import numpy as np 

raw_data = pd.read_csv("blooddata.txt", header = False, sep = ",")

test_ind = np.random.randint(0, 746, size= 149)

test_set = raw_data.ix[test_ind]

train_ind = set(range(747)).difference(set(test_ind))

train_set = raw_data.ix[list(train_ind)]

returning = train_set.groupby("Donated_2007")

not_ret = train_set.ix[returning.groups[0]]
ret = train_set.ix[returning.groups[1]]

import matplotlib.pyplot as plt 

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(ret['Time_Last_Donation'], ret['Donated_2007'], s = 10, marker="s", label="returning")
ax1.scatter(not_ret['Time_Last_Donation'], not_ret['Donated_2007'], s = 10, marker="o", label="not returning")
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(111)

ax2.scatter(ret['Monetary'], ret['Donated_2007'], s = 10, marker="s", label="returning")
ax2.scatter(not_ret['Monetary'], not_ret['Donated_2007'], s = 10, marker="o", label="not returning")
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
ax3 = fig.add_subplot(111)

ax3.scatter(ret['Frequency'], ret['Donated_2007'], s = 10, marker="s", label="returning")
ax3.scatter(not_ret['Frequency'], not_ret['Donated_2007'], s = 10, marker="o", label="not returning")
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
ax4 = fig.add_subplot(111)

ax4.scatter(ret['Recency'], ret['Donated_2007'], s = 10, marker="s", label="returning")
ax4.scatter(not_ret['Recency'], not_ret['Donated_2007'], s = 10, marker="o", label="not returning")
plt.legend(loc='upper left')
plt.show()

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.ensemble import GradientBoostingClassifier

forest = RandomForestClassifier(n_estimators = 100)

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

param_dist = {"max_depth": [4, None],
			  "max_features": sp_randint(1,2),
			  "min_samples_split": sp_randint(1,2),
			  "min_samples_leaf": sp_randint(1,2),
			  "bootstrap": [True, False],
			  "criterion": ["gini", "entropy"]}

random_search = RandomizedSearchCV(forest, param_distributions=param_dist, n_iter=20)


random_search.fit(train_set[["Recency", "Frequency", "Monetary", "Time_Last_Donation"]], train_set["Donated_2007"])

report(random_search.grid_scores_)

bestmodel = RandomForestClassifier(n_estimators=100, criterion="entropy", max_features=1, bootstrap=True, max_depth=4, min_samples_leaf=1, min_samples_split=1)

forest_fit = bestmodel.fit(train_set[["Recency", "Frequency", "Monetary", "Time_Last_Donation"]], train_set["Donated_2007"])

result = forest_fit.predict(test_set[["Recency", "Frequency", "Monetary", "Time_Last_Donation"]]) # error: 24.8%
true_test_classes = test_set["Donated_2007"]

np.sum(abs(true_test_classes - result))/len(result) # better! 0.20805 

gradient = GradientBoostingClassifier(n_estimators=200)

param_dist_grad = {"loss": ["deviance", "exponential"],
                   "learning_rate": [0.05, 0.1, 0.2],
                   "max_depth": [1,2,3],
                   "max_leaf_nodes": None,
                   "max_features": sp_randint(1,4)}

random_search_grad = RandomizedSearchCV(gradient, param_distributions=param_dist_grad, n_iter=20)

random_search_grad.fit(train_set[["Recency", "Frequency", "Monetary", "Time_Last_Donation"]], train_set["Donated_2007"])

report(random_search_grad.grid_scores_)



