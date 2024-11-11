# K-Nearest-Neighbors-KNN-for-Classification

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Create a dataset
X = np.array([[1, 2], [2, 3], [3, 3], [6, 8], [7, 8], [8, 9]])
y = np.array([0, 0, 0, 1, 1, 1])  # Labels for two classes

# Define KNN function
def knn(X_train, y_train, x_test, k):
    distances = [np.sqrt(np.sum((x - x_test) ** 2)) for x in X_train]
    nearest = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest]
    return Counter(nearest_labels).most_common(1)[0][0]

# Predict class for a new point
x_new = np.array([5, 5])
predicted_class = knn(X, y, x_new, k=3)
print("Predicted class:", predicted_class)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
plt.scatter(x_new[0], x_new[1], c='green', marker='*', s=200, label="New Point")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("K-Nearest Neighbors Classification")
plt.show()
