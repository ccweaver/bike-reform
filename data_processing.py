import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB

raw_file = open('../2014-07-CitiBikeTripData.csv','rb')
#raw_file = open('../subset.csv')
weather_file = open('../julyWeather.csv', 'rU')

test_percentage = 10
print str(test_percentage) + "% Testing, Features: Start Station ID, Start Hour"

test_x = []
test_y = []
train_x = []
train_y = []

raw_data = np.genfromtxt(raw_file, dtype=str, delimiter=',', skip_header=1)
x_data_list = [[int(r[3][1:-1]),int(r[1].split(' ')[1].split(":")[0])] for r in raw_data]
y_data_list = [int(r[7][1:-1]) for r in raw_data]

mean_temps = []
precipitation = []
for line in weather_file:
	entries = line.split(',')
	mean_temps.append(int(entries[2]))
	precipitation.append(float(entries[4]))

# get total number of rides at each group of temperature and precipitation values
num_rides_temp = np.array([0,0,0,0])
num_rides_precip = np.array([0,0,0,0,0])

for r in raw_data:
	date = int(r[1][9:11]) - 1
	# count the number of rides in each range of 5 degrees
	#print "temp " + str(len(num_rides_temp))
	#print "perc " + str(len(num_rides_precip))
	#print str(date)
	if mean_temps[date] < 75:
		num_rides_temp[0] += 1
	elif mean_temps[date] < 80:
		num_rides_temp[1] += 1
	elif mean_temps[date] < 85:
		num_rides_temp[2] += 1
	else:
		num_rides_temp[3] += 1

	# do the same for precipitation
	if precipitation[date] == 0:
		num_rides_precip[0] += 1
	elif precipitation[date] < 0.5:
		num_rides_precip[1] += 1
	elif precipitation[date] < 1.0:
		num_rides_precip[2] += 1
	elif precipitation[date] < 1.5:
		num_rides_precip[3] += 1
	else:
		num_rides_precip[4] += 1

print num_rides_temp

fig = plt.figure()
ax = fig.add_subplot(111)

temp_indices = np.arange(4)#['<75','75-80','80-85','85-90','90+']
temp_graph = ax.bar(temp_indices, num_rides_temp, width=0.5, color='r', label='Temperature')

ax.set_title('Number of Rides as a Factor of Temperature')
ax.set_ylabel('Number of Rides')
ax.set_xlim(70, 90)
ax.set_ylim(0, 100)

xTickMarks = ['<75', '75-80', '80-85', '85+']
xTickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xTickNames)
plt.show()

#precip_indices = np.arange(5)#['0','0.01-0.5','0.5-1.0','1.0-1.5','1.5+']
#precip_graph = plt.bar(precip_indices, num_rides_precip, color='b', label='Precipitation')


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

#print "X shape: ", train_x.shape, "y shape: ", train_y.shape
#print "Test X shape: ", test_x.shape, "Test y shape: ", test_y.shape
#print len(stations)
#num_stations = len(stations)

# print "MultiClass Gaussian Naive Bayes Classification of End Station"
# start = time.time()
# clf = OutputCodeClassifier(GaussianNB(), code_size=2, random_state=0)
# fit = clf.fit(train_x, train_y)
# end = time.time()
# print "Fit time: ", end-start
# start = time.time()
# y_preds = fit.predict(test_x)
# end = time.time()
# print "Predict time: ", end-start

# correct = 0
# for pred, truth in zip(y_preds, test_y):
# 	if pred == truth:
# 		correct += 1
# accuracy = float(correct)/len(y_preds)
# print accuracy



