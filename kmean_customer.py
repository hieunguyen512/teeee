import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # Ngăn việc mở GUI
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
df = pd.read_csv(filename, low_memory=False)

# df = pd.read_csv('Mall_Customers.csv', low_memory=False)
# X là Annual Income và	Spending Score
X = df.iloc[:, [3, 4]].values
# X là Age và Spending Score
Y = df.iloc[:, [2, 4]].values
# Dự đoán k
# Sẽ cập nhập

print(X)
#Huấn luyện mô hình K-Means với số tâm cụm tối ưu k = 5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
#Dự đoán với X


y_kmeans = kmeans.fit_predict(X)
x_cluster = kmeans.cluster_centers_
#Huấn luyện mô hình K-Means với số tâm cụm tối ưu k = 4
kmeans2 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
#Dự đoán với X
y_pred = kmeans2.fit_predict(Y)
y_cluster = kmeans2.cluster_centers_

## Vẽ 1 đồ thị phân cụm:
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()

# Vẽ 2 đồ thị subplots:
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,10))
# subplot 1:
# Vẽ các điểm
ax1.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
ax1.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
ax1.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
ax1.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
ax1.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# Vẽ tâm cụm
ax1.scatter(x_cluster[:, 0], x_cluster[:, 1], s = 300, c = 'yellow',marker='*', label = 'Centroids')
ax1.set(title="Spending Score By Asset",
        xlabel='Annual Income (k$)',
        ylabel='Spending Score')
ax1.legend()
# subplot 2:
# Vẽ các điểm
ax2.scatter(Y[y_pred == 0, 0], Y[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
ax2.scatter(Y[y_pred == 1, 0], Y[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
ax2.scatter(Y[y_pred == 2, 0], Y[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
ax2.scatter(Y[y_pred == 3, 0], Y[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# Vẽ tâm cụm
ax2.scatter(y_cluster[:, 0], y_cluster[:, 1], s = 300, c = 'yellow',marker='*', label = 'Centroids')
ax2.set(
        xlabel='Age',
        ylabel='Spending Score')
ax2.legend()
plt.show()