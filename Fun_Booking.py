
def Fun(x1, x2, x3, x4):
	while True:
		if x3 > 0:
			x3 = x3 - 1
			x4 = x4 + 1
		elif x2 > 0:
			x2 = x2 - 1
			x3 = x3 + x4
		elif x1 > 0:
			x1 = x1 - 1
			x2 = x2 + x4
			x4 = 1
		else:
			return x4
