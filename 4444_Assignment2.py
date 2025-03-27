import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

# Train KNN with K=7
k = 7
clf = neighbors.KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# Decision boundary plot
h = .02  
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title(f"Decision Boundary for KNN (k={k})")
plt.show()
