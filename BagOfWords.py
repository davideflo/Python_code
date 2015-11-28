import pandas as pd 

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)

train.shape
train.columns.values

from bs4 import BeautifulSoup

example1 = BeautifulSoup(train["review"][0])

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

words = [w for w in words if not w in stopwords.words("english")]

def review_to_words(raw_review):
	review_text = BeautifulSoup(raw_review).get_text()
	letters_only = re.sub("^[a-zA-Z]", " ", review_text)
	words = letters_only.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = [w for w in words if not w in stops]
	return(" ".join(meaningful_words))

num_reviews = train["review"].size
clean_train_reviews = []
for i in range(0, num_reviews):
	clean_train_reviews.append(review_to_words(train["review"][i]))

print("creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,
	preprocessor = None, stop_words = None, max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

import numpy as np 

train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
print(vocab)

dist = np.sum(train_data_features, axis = 0)

for tag, count in zip(vocab, dist):
	print(tag, count)

# find the maximum/minimum
#
# L = []
# for tag, count in zip(vocab, dist):
#	L.append([tag, count])
#
# l = sorted(L, key = lambda a: a[1])
#
# max = l[len(l) - 1]

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features, train["sentiment"])

result = forest.predict(train_data_features)
















