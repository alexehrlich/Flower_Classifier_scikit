from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from scipy.stats import mode

def map_clusters(ground_truth, predicted_labels) -> np.array:
	"""Mapping clusters to the truth data becaus kmeans 
	create random clusterssince it is unlabeld data"""

	label_mapping = {}
	for cluster in np.unique(predicted_labels):
		#Foor every cluster create a mask vector with True False
		# which is True wherever the prediction set the current clsuter
		# in cluster_labels. With that we mapped the prediction
		mask = predicted_labels == cluster

		#We access the the acutal values where the mask is true. So we see what
		# acutal values there should be instead of what our clustering set
		true_labels = ground_truth[mask]

		#Access the most frequent label to be the correct one and safe it
		# in order to map it to "wrong" prediction
		label_mapping[cluster] = mode(true_labels)[0]
	
	#set the correct labels with the mapped ones
	return np.array([label_mapping[label] for label in predicted_labels])


def main():
	"""
		Although IRIS is labeled we use it unlabeled to explore Kmeans.
	"""
	iris = load_iris()

	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	X = iris.data
	y = iris.target
	print(df.head())

	scaler = StandardScaler()
	X_std = scaler.fit_transform(X)

	#Show scatter plots between all features and histogram
	sns.pairplot(df)
	plt.show()

	#Ususally the elbow method is used to determine the number of clusters
	# Here we want 3 clusters since we know there are 3 species
	kmeans = KMeans(n_clusters=3, random_state=10)
	kmeans.fit(X_std)
	cluster_labels = kmeans.labels_

	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X_std)
	plt.figure(figsize=(8, 6))
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', alpha=0.7)
	plt.title("K-Means Clustering Results")
	plt.xlabel("PCA Component 1")
	plt.ylabel("PCA Component 2")
	plt.colorbar(label="Cluster")
	plt.show()


	mapped_clusters = map_clusters(y, cluster_labels)

	cm = confusion_matrix(y, mapped_clusters)
	disp = ConfusionMatrixDisplay(cm)
	disp.plot()
	plt.show()


if __name__ == '__main__':
	main()
