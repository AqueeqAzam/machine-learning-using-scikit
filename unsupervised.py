# k-means cluster
# Step-1: Select the value of K, to decide the number of clusters to be formed.
# Step-2: Select random K points which will act as centroids.
# Step-3: Assign each data point, based on their distance from the randomly selected points
# (Centroid), to the nearest/closest centroid which will form the predefined clusters.
# Step-4: place a new centroid of each cluster.
# Step-5: Repeat step no.3, which reassign each datapoint to the new closest centroid of each cluster.
# tep-6: If any reassignment occurs, then go to step-4 else go to Step 7.
# Step-7: FINISH

# importing modules
from sklearn import datasets
import matplotlib.pyplot as plt

# loading data
"""iris_df = datasets.load_iris()

# Available methods on datasets
print(iris_df)

# feature
print(iris_df.feature_names)

# target
print(iris_df.target)

# target names
print(iris_df.target_names)
label = {0: 'red', 1:'blue', 2:'green'}

# dataset slicing
x_axis = iris_df.data[:, 0]
y_axis = iris_df.data[:, 2]

# plotting
plt.scatter(x_axis, y_axis, c= iris_df.target)
plt.show()



#  clustering -> Data divided into several group with similar traits.
# k means cluster
from sklearn import datasets
from sklearn.cluster import KMeans

# loading data
iris_df = datasets.load_iris()

# declaring model
model = KMeans(n_clusters=4)

# fitting model
model.fit(iris_df.data)

# predict the single input
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# predict on the entire data
all_predictions = model.predict(iris_df.data)

# printing prediction
print(predicted_label)
print(all_predictions)"""


# hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

# reading the dataframe
seed_df = pd.read_csv('https://raw.githubusercontent.com/vihar/unsupervised-learning-with-python/master/seeds-less-rows.csv')

# remove the grain species from the dataframe, sava for later
varieties = list(seed_df.pop('grain_variety'))

# extract the measurement as a numpy array
samples = seed_df.values

margins = linkage(samples, method = 'complete')

dendrogram(margins, labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

