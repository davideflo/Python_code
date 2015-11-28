# Stack Exchange Question Classifier

import pandas as pd
import numpy as np
import json

import codecs

def read_dataset(path):
  with codecs.open(path, 'r', 'utf-8') as myFile:
    content = myFile.read()
  dataset = json.loads(content)
  #dataset = pd.read_table(content)
  return dataset



train = pd.read_table("input00.txt", header = 0, delimiter = "\t", quoting = 3)


label = open("output00.txt", 'r')
test_num_classes = []

for line in label:
	test_num_classes.append(str(line))

test_indices = np.random.random_integers(0, train.shape[0]-1, 1000)

train_indices = set(range(train.shape[0])).difference(set(test_indices))

Train = train.ix[train_indices]


Train_classes = []
for index in train_indices:
	Train_classes.append(test_num_classes[index])



Test = train.ix[test_indices]

Test_classes = []
for index in test_indices:
	Test_classes.append(test_num_classes[index])


from bs4 import BeautifulSoup

example1 = BeautifulSoup(str(Train.ix[0]))




import re

letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())

# [] indicates group membership and ^ means "not". In other words, the re.sub() statement above says, "Find anything that is NOT a lowercase letter (a-z) or an upper case letter (A-Z), and replace it with a space."

lower_case = letters_only.lower()
words = lower_case.split()

import nltk
#nltk.download()
# look into some NLP documentation
from nltk.corpus import stopwords # import the stop words list

print(stopwords.words("english"))

def review_to_words(raw_review):
	review_text = BeautifulSoup(str(raw_review)).get_text()
	letters_only = re.sub("^[a-zA-Z]", " ", review_text)
	words = letters_only.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	return(" ".join(meaningful_words))

true_words = []
for i in train_indices:
	true_words.append(review_to_words(Train.ix[i].values))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
	preprocessor = None, stop_words = None, max_features = 6000)

train_data_features = vectorizer.fit_transform(true_words)

train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

num_classes = []
for x in Train_classes:
	if x == 'electronics':
		num_classes.append(0)
	elif x == 'mathematica':
		num_classes.append(1)
	elif x == 'android':
		num_classes.append(2)
	elif x == 'security':
		num_classes.append(3)
	elif x == 'gis':
		num_classes.append(4)
	elif x == 'photo':
		num_classes.append(5)
	elif x == 'scifi':
		num_classes.append(6)
	elif x == 'unix':
		num_classes.append(7)
	elif x == 'apple':
		num_classes.append(8)
	else:
		num_classes.append(9)	
	
	


classifier = rfc.fit(train_data_features, num_classes)

test_words = []
for i in test_indices:
	test_words.append(review_to_words(Test.ix[i].values))

test_data_features = vectorizer.fit_transform(test_words)

test_data_features = test_data_features.toarray()

result = classifier.predict(test_data_features)
result2 = classifier.predict(train_data_features)