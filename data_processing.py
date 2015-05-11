import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB

raw_file = open('../2013-07-CitiBikeTripData.csv','rb')
#raw_file = open('../subset.csv')

test_percentage = 10
print str(test_percentage) + "% Testing, Features: Start Station ID, Start Hour"

test_x = []
test_y = []
train_x = []
train_y = []

raw_data = np.genfromtxt(raw_file, dtype=str, delimiter=',', skip_header=1)
x_data_list = [[int(r[3][1:-1]),int(r[1].split(' ')[1].split(":")[0]),int(r[0][1:-1]),1 if (r[8] == "Customer") else 0] for r in raw_data]
y_data_list = [int(r[7][1:-1]) for r in raw_data]

stations = []
i = 0
for x, y in zip(x_data_list, y_data_list):
	if i%test_percentage == 0:
		test_x.append(x)
		test_y.append(y)
	else:
		train_x.append(x)
		train_y.append(y)
	if y not in stations:
		stations.append(y)
	i+=1
train_x = np.asarray(train_x, dtype=int)
train_y = np.asarray(train_y, dtype=int)
test_x = np.asarray(test_x, dtype=int)
test_y = np.asarray(test_y, dtype=int)

print "X shape: ", train_x.shape, "y shape: ", train_y.shape
print "Test X shape: ", test_x.shape, "Test y shape: ", test_y.shape
print len(stations)
num_stations = len(stations)

print "MultiClass Gaussian Naive Bayes Classification of End Station"
start = time.time()
clf = OutputCodeClassifier(GaussianNB(), code_size=2, random_state=0)
fit = clf.fit(train_x, train_y)
end = time.time()
print "Fit time: ", end-start
start = time.time()
y_preds = fit.predict(test_x)
end = time.time()
print "Predict time: ", end-start

correct = 0
for pred, truth in zip(y_preds, test_y):
	if pred == truth:
		correct += 1
accuracy = float(correct)/len(y_preds)
print accuracy

