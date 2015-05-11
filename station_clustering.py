import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans

# raw_file = open('../2013-07-CitiBikeTripData.csv','rb')

# raw_data = np.genfromtxt(raw_file, dtype=str, delimiter=',', skip_header=1)

# origin_stations = [int(r[3][1:-1]) for r in raw_data]
# sta_max = 0
# for x in origin_stations:
# 	if x > sta_max:
# 		sta_max = x


# trip_counts = np.zeros((sta_max,24))

# for r in raw_data:
# 	trip_counts[int(r[3][1:-1])-1][int(r[1].split(' ')[1].split(":")[0])] += 1

# trip_counts = [[float(x) for x in r] for r in trip_counts]
# trip_counts = np.asarray(trip_counts)

# np.save("tripcounts.npy",trip_counts)

trip_counts = np.load("tripcounts.npy")
t = []
for station in trip_counts:
	st = np.std(station)
	s = np.zeros(len(station)) - 1
	if st != 0:
		s = (station - np.mean(station))/st
	t.append(s)
trip_counts = np.asarray(t)




print "K means clustering of number of station activity by hour"
start = time.time()
kmeans =  KMeans(n_clusters=20,random_state=0)
fit = kmeans.fit(trip_counts)
end = time.time()
print "Clustering time: ", end-start

clusts = fit.cluster_centers_

fig = plt.figure(figsize=(12, 8))
plt.imshow(clusts,interpolation='none')
plt.xlabel('Hour')
plt.ylabel('Cluster Index')
plt.title('Cluster Trip Count Heat-Map')
plt.show()



