### Majority

import numpy as np 
import pandas as pd 
import math


def Initialize_Balls(N, prop):
	balls = []
	for trials in range(N):
		balls.append(np.random.binomial(1, prop))
	return balls

def CheckColor(i,j, balls):
	if abs(balls[i] - balls[j]) == 0:
		return 1
	else:
		return 0

def CheckColorConsecutives(i, j, k, balls):
	if abs(CheckColor(i,j, balls) - CheckColor(j,k, balls)) == 0:
		return 1
	else:
		return 0

def FindMajority(balls):
	N = len(balls)
	trials = 1
	same_color_0 = set([])
	same_color_1 = set([])
	if CheckColor(0,1,balls) == 1:
		same_color_0.add(0)
		same_color_0.add(1)
	else:
		same_color_0.add(0)
		same_color_1.add(1)
	for i in range(2,N):
		if CheckColor(1, i, balls) == 1:
			same_color_0.add(i)
		else:
			same_color_1.add(i)
		trials += 1
		if trials >= math.ceil(N/2):
			if len(same_color_0) > len(same_color_1):
				return same_color_0
			else: 
				return same_color_1







