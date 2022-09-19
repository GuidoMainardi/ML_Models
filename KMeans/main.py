from Kmeans import KMeans
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('column_data.csv')
print(data.head())

dists = []
for i in range(1, 10):
    print(i)
    km = KMeans(n_clusters = i, tries=50)
    km.fit(data.drop(['class'], axis=1))
    dists.append(km._clusters_variance())

plt.figure()
plt.plot(range(1, 10), dists, 'bx-')
plt.xlabel('n clusters')
plt.ylabel('total variance')
plt.title('elbow method')
plt.show()