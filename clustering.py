import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB
import lda

raw_file = open('../bike_data/2013-07-CitiBikeTripData.csv','rb')
# raw_file = open('../bike_data/subset.csv')

test_percentage = 10
print str(test_percentage) + "% Testing, Features: Start Station ID, Start Hour"

test_x = []
test_y = []
birthyears = []
genders = []

raw_data = np.genfromtxt(raw_file, dtype=str, delimiter=',', skip_header=1)

for i in range(len(raw_data)):
	if raw_data[i][13] != '\N':
		birthyears.append(int(raw_data[i][13][1:-1]))
		genders.append(int(raw_data[i][14][1:-1]))



# raw_data = np.genfromtxt(raw_file, dtype=str, delimiter=',', skip_header=1)
# x_data_list = [int(r[3][1:-1]) for r in raw_data]
# y_data_list = [int(r[7][1:-1]) for r in raw_data]

# stations = []
# i = 0
# for x, y in zip(x_data_list, y_data_list):
# 	if i%test_percentage == 0:
# 		test_x.append(x)
# 		test_y.append(y)
# 	else:
# 		train_x.append(x)
# 		train_y.append(y)
# 	if y not in stations:
# 		stations.append(y)
# 	i+=1

# train_x = np.asarray(train_x, dtype=int)
# train_y = np.asarray(train_y, dtype=int)
# test_x = np.asarray(test_x, dtype=int)
# test_y = np.asarray(test_y, dtype=int)

# print train_x;
# print train_y;

# zipped = zip(train_x, train_y);

# maxstation = max(max(train_x), max(train_y)) + 1;
# trips = [[0]*maxstation for _ in range(maxstation)];

# for x, y in zipped:
# 	trips[x][y] += 1;
# 	trips[y][x] += 1;

# nonzeros = 0;
# for i in range(maxstation):
# 	for j in range(maxstation):
# 		if (trips[i][j] != 0):
# 			nonzeros += 1;

# sorted(trips);