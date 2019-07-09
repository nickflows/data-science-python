# Applied Machine Learning

## Week 1 

### Supervised Learning: Learn to predict target values with label data

- Classification: Target values are discrete (learn a classifier function)
- Regression: Target values are continuous (learn a regression function)
- Label Data:
	- Explicit Data (e.g. hand-labeled by a human)
	- Implicit Data (e.g. clicks)



### Unsupervised Learning: Find structure in unlabeled data

- Find groups or clusters of data (clustering)
- Finding unusual patterns (outlier detection)
  

### A basic machine learning workflow (Classification as an example)

- Representation: Choose a feature representation and a model to use
	- Input to the learning function
- Evaluation: What criteria and metrics distinguish a good performing classfier from a bad performing one?
	- quality or accuracy score
- Optimization: Searching for the best parameters and settings in a model

### Code Snippets


- Creating a training and test set:
	- _import library:_ `from sklearn.model_selection import train_test_split`
	- _create tuples:_ `lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))`
	- _create dataset:_ `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)`
	- Create 4 variables and splits the data into training and test

- Training Data Visualization
	- Range of values (find outliers or noise in the dataset)
	- Determine likelihood that a machine learning algo can classify dataset (how well clustered are the datasets)
		- Looking at feature space

- "Feature-pair plot" Code Snippet 
	- Shows how the features are correlated or not
	- Useful for smaller datasets
	- `cmap = cm.get_cmap('gnuplot')`
	- `scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)`


- plotting a 3D scatter plot
	-_Import plotting library:_ `from mpl_toolkits.mplot3d import Axes3D`
-_plot 3d figure using a scatter plot:_
	`fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
	ax.set_xlabel('width')
	ax.set_ylabel('height')
	ax.set_zlabel('color_score')
	plt.show()`


### K-Nearest Neighbors

- KNN Classifiers - Memory / Instance Based Supervised Learning
	- Memorization of labels in training set
- "K" represents the number of nearest neighbors it will retrieve and use to make its prediction
- Three Steps to nearest neighbor algorithm
	- Find the `K` most similar instances of X that was in the training set
	- Get those labels for the similar instances in the training set
	- Combine the labels to make a prediction for X (e.g. simple majority vote)
- A nearest Neighbor algorithm needs four things (four requirements for KNN algorithm)
	- A distance metric (e.g. Euclidean Distance)
	- How many nearest neighbors to use (K)
	- Optional weighting function on the neighbor points
	- Aggregation method for nearest neighbor points
	
#### Code for KNN Classifier

_import library_ `from sklearn.neighbors import KNeighborsClassifier`
		
## Week 2
