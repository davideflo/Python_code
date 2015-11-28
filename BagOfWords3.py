# Bag Of Words part 3

from gensim.models import Word2Vec
import numpy as np 
import pandas as pd

model = Word2Vec.load("300features_40minwords_10context") 

def makeFeatureVec(words, model, num_features):
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0.
	index2word_set = set(model.index2word)
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec, model[word])
	featureVec = np.divide(featureVec, nwords)
	return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
	counter = 0.
	reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = "float32")
	for review in reviews:
		reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
		counter = counter + 1.
	return reviewFeatureVecs

clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append(review_to_wordlist(review, remove_stopwords = True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append(review_to_wordlist(review, remove_stopwords = True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(trainDataVecs, train["sentiment"])
result = forest.predict(testDataVecs)

output = pd.DataFrame(data = {"id":test["id"], "sentiment":result})

from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = int(word_vectors.shape[0] / 3)


kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

end = time.time()
elapsed = end - start

word_centroid_map = dict(zip(model.index2word, idx))

for cluster in range(0,10):
	print("\nCluster %d " % cluster)
	words = []
	list_of_values = list(word_centroid_map.values())
	list_of_keys = list(word_centroid_map.keys())
	for i in range(0, len(word_centroid_map.values())):
		if(list_of_values[i] == cluster):
			words.append(list_of_keys[i])
	print(words)

def create_bag_of_centroids(wordlist, word_centroid_map):
	num_centroids = max(word_centroid_map.values())+1
	bag_of_centroids = np.zeros(num_centroids, dtype="float32")
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1
	return bag_of_centroids

train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_train_reviews:
	train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
	counter += 1

test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
	test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
	counter += 1

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

output = pd.DataFrame(data = {"id": test["id"], "sentiment": result})

















