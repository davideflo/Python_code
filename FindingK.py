import numpy as np 
import pandas as pd 
import random as rd 

def Wk(mu, clusters):
	K = len(mu)
	return sum([np.linalg.norm(mu[i] - c)**2/(2*len(c)) for i in range(K) for c in clusters[i]])

def bounding_box(X):
	xmin, xmax = min(X, key = lambda a:a[0])[0], max(X, key = lambda a: a[0])[0]
	ymin, ymax = min(X, key = lambda a:a[1])[1], max(X, key = lambda a:a[1])[1]
	return ((xmin, xmax), (ymin, ymax))

def cluster_points(X, mu):
	clusters = {}
	for x in X:
		bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
		try:
			clusters[bestmukey].append(x)
		except KeyError:
			clusters[bestmukey] = [x]
	return clusters

def reevaluate_centers(mu, clusters):
	newmu = []
	keys = sorted(clusters.keys())
	for k in keys:
		newmu.append(np.mean(clusters[k], axis = 0))
	return newmu

def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
	oldmu = rd.sample(X, K)
	mu = rd.sample(X, K)
	while not has_converged(mu, oldmu):
		oldmu = mu
		clusters = cluser_points(X, mu)
		mu = reevaluate_centers(oldmu, clusters)
	return(mu, clusters)

def gap_statistic(X):
	(xmin, xmax), (ymin, yman) = bounding_box(X)
	ks = range(1,10)
	Wks = zeros(len(ks))
	Wkbs = zeros(len(ks))
	sk = zeros(len(ks))
	for indk, k in enumerate(ks):
		mu, clusters = find_centers(X, k)
		Wks[indk] = np.log(Wk(mu, clusters))
		B = 10
		BWkbs = zeros(B)
		for i in range(B):
			Xb = []
			for n in range(len(X)):
				Xb.append([rd.uniform(xmin, xmax), rd.uniform(ymin, ymax)])
			Xb = np.array(Xb)
			mu, clusters = find_centers(Xb, k)
			BWkbs[i] = np.log(Wk(mu, clusters))
		Wkbs[indk] = sum(BWkbs)/B 
		sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk])**2)/B)
	sk = sk*np.sqrt(1 + 1/B)
	return(ks, Wks, Wkbs, sk)