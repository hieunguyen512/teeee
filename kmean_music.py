import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # Ngăn việc mở GUI
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
df = pd.read_csv(filename, low_memory=False)

# dataset = pd.read_csv('Mall_Customers.csv', low_memory=False)

# Chọn 2 feature phù hợp
# X là energy và valance
X = df.iloc[:,[1,9]].values

# Dự đoán k
# Sẽ cập nhập

#Huấn luyện mô hình K-Means với số tâm cụm tối ưu k = 5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
#Dự đoán với X
y_kmeans = kmeans.fit_predict(X)

#Vẽ các điểm
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red',alpha=0.3, label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue',alpha=0.3, label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green',alpha=0.3, label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'Gray',alpha=0.3, label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'pink',alpha=0.3, label = 'Cluster 5')
#Vẽ tâm
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow',marker='*', label = 'Centroids')
plt.title('Clusters of music')
plt.xlabel('Energy Point')
plt.ylabel('Valance Point')
plt.legend()
plt.show()
