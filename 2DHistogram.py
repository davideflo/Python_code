import pandas as pd 

me = pd.ExcelFile('meteorites-updated_05-14-13.xlsx')

data = me.parse()

def find_range_long(lo, long):
	s = np.shape(long)[0]
	loc = np.zeros(s)
	for l in lo:
		for i in range(s-1):
			if (l >= long[i]) & (l < long[i+1]):
				loc[i] += 1
			else:
				loc[24] +=1
	return loc

def find_range_long2(df, long):
	s = np.shape(long)[0]
	loc = np.zeros(s)
	loc[-1] = np.shape(df.ix[df['reclong'] >= 165])[0] 
	for i in range(s-1):
		loc[i] += np.shape(df.ix[(df['reclong'] >= long[i]) & (df['reclong'] < long[i+1])])[0]
	return loc

def compute_histogram(df):
	longi = np.arange(-180, 180, 15)
	lat = np.arange(-90, 90, 15)
	slo = np.shape(longi)[0]
	sla = np.shape(lat)[0]
	hist = np.zeros(shape=(sla, slo))
	for i in range(sla):
		for j in range(slo):

