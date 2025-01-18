# IRIS

In this simple demo I use the IRIS data set to explore Pytorch and Scikit-learn.

## KNearestNeighbor
Uses PCA to reduse the 4 dimensional input vecor [sepal width, sepal length, petal width, petal length] to a 2 dimensional vector using PCA to make a scatter plot to easily make the clusters visible.

## Feed Forward Neural Network
Trains a model with one hidden layer. Moves the tensors to the GPU for faster execution.

## Kmeans
Although IRIS is labeled data we can use it to explore Kmeans and pretend it as unlabeled
data. We look at the scatter plots between all four features, we reduce it to 2 dimensions with PCA to plot it and in the end we evaluate it with the confusion matrix.

The biggest learning here is, that the result highly depends on the random_sate where the cluster centroids are set in  the beginning.


## Usage
- `make setup`to install the dependencies
- `make KNN`to run the K Nearest Neighbor Classification
- `make FNN` to train a fully connected Feed Forward Neural Network
- `make KMEAN` to run the Kmeans to cluster the data
