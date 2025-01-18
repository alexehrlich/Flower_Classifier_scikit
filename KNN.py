import numpy as np
import pandas as pd
import matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

def main():

    iris_dataset = datasets.load_iris()

    #data contains the feature Matrix
    df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)

    #Add the species label
    # target is the target vector matching the feature matrix 
    df['species'] = pd.Categorical.from_codes(iris_dataset.target, iris_dataset.target_names)

    print(df.head())

    #Split the Data into training and test data
    X = iris_dataset.data
    y = iris_dataset.target

    #Standardize the data for better convergence
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    #Prinicpal componentn Analysis of the training set to plot, Reduced to 2 dimension
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_standardized)

    # Plot PCA-reduced data
    # y(vector) == i (int) returns a Bool array of the len of the data.
    # With that we access only the target we want.
    plt.figure(figsize=(8, 6))
    for i, target_name in enumerate(iris_dataset.target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA of Iris Dataset')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = knn.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()