Optional - Unsupervised Learning

Unsupervised Learning involves tasks that operate on datasets without labeled responses or target values. The goal is to capture the structure and information.

Some applications include:
- Visualize structures of complex datasets
- Density estimation to predict probabilities of events
- Compress and summarize the data
- Extract features for supervised learning
- Discover important clusters or outliers


Two major categories of unsupervised methods:
- Transformation: extracts or computes information of some kind
- Clustering: find groups of data, and assign data points to a group


## Transformation
* Density Estimation - Probability of observing a measurement at a specific point
	- Kernal density class to do kernal density estiamtion
	- Distribution over likelihood of a class
* 



## Dimensionality Reduction
- Finds an approximate version of your dataset using fewer features
- Also used for compression, finding features for supervised learning

### Principal Component Analysis
- Takes original data points and finds rotation s.t. dimensions are uncorrelated
- Drops all but the most informative dimensions (e.g. most of variation is captured by these dimensions)

*Code Snippet*
```
from sklearn.decomposition import PCA
# Before applying PCA, each feature should be centered (zero mean) and with unit variance
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  

pca = PCA(n_components = 2).fit(X_normalized)
X_pca = pca.transform(X_normalized)
```

### Manifold Learning
- Very good at finding low-dimensional structure in high-dimensional data
- Manifold: low dimensional sheet in a high dimensional space
- Multi-dimension scaling: visualize a high-dimensional dataset in a low-dimensional space (e.g. 2d or 3d space)
- t-SNE: a powerful manifold learning method that finds a 2D projection, preserving information about neighbors 


## Clustering
- Data points within the same cluster should be "close" or "similar" in some way
- Data points are either assigned a likelihood score for their cluster assignment or just a cluster


### K-Means Clustering
- Pick the number of clusters, `k`, you want to find. Pick `k` random points to start as the center for each cluster.
- Step 1: Assign each data point to the nearest cluster center
- Step 2: Update each cluster center by replacing it with the mean of all points assigned to the cluster
- Repeat steps #1 and #2 until the measn converge

*Notes on K-Means:*
- Very sensitive to the range of values, so likely need to normalize feature values
- Works well for simple clusters, but not on irregular clusters
- k-means works well for continuous features. There are variants of k-means that can work with categorical variables


*Code Snippet*
```
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

fruits = pd.read_table('fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']].as_matrix()
y_fruits = fruits[['fruit_label']] - 1

X_fruits_normalized = MinMaxScaler().fit(X_fruits).transform(X_fruits)  

kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(X_fruits_normalized)
```


### Agglomerative Clusters
- Family of clustering methods that perform a set of iterative, bottoms-up approach
- Can specify linkage criteria:
	- Ward's Method: Merge the 2 clusters that given the smallest increase in total variance
	- Average: Merge 2 clusters that have the smalled average distance between points
	- Complete: Merge 2 clusters that have the smallest maximum distance between points
- Ward's is usually the best method.


*Code Snippet*
```
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from adspy_shared_utilities import plot_labelled_scatter

X, y = make_blobs(random_state = 10)

cls = AgglomerativeClustering(n_clusters = 3)
cls_assignment = cls.fit_predict(X)

plot_labelled_scatter(X, cls_assignment, 
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])
```

### DBSCAN
- Density Based Spacial Clustering of Applications with Noise
- Benefits
	- Don't need to specify the number of clusters
	- It can find outlier data points
	- Relatively efficient, and can be used with large datasets
- Parameters:
	- `eps`
	- `min_samples`





