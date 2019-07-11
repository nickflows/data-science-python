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

- _import library_ `from sklearn.neighbors import KNeighborsClassifier`
- _assign classifier object to a variable:_ `knn = KNeighborsClassifier(n_neighbors = 1)`
- _fit the model to a train and test dataset:_ `knn.fit(X_train, y_train)`
- _predict values on a test dataset_ `knn.predict(X_test)`
- _produce accuracy score on test data:_ `knn.score(X_test, y_test)`
		
## Week 2 - Introduction to Supervised Machine Learning

### Classificaton & Regression


Both classification and regressiong take a set of training instances (values) and learn a mapping to a target value
- for classification, the target value is a discrete class value
-- Binary classification: two classes, a "negative class" and "positive class"
-- Multi-class: target value is a set of discrete values
-- Multi-label classification: multiple target values (training labels)

For Regression, the target value is continuous (floating point / real-valued)
- Example: predicting the price for a house



Focus on two types of algorithms:
- K-nearest neighbors: makes few assumptions about the structure of the data, and give potentially accurate but sometimes unstable results.
- Linear model fit using least-squares: makes strong assumptions about the structure of data, and gives stable but potentially inaccurate results.


### Generalization, Overfitting, and Underfitting

- Generalization - algorithm's ability to give accurate predictions for new, previously unseen data
- Assumptions
  -- Future unseen data (test set) will have the same properties as the current training sets (drawn from the same population)


- Models that are too complex for the amount of training data available are said to "overfit" and are not likely to generalize to new examples
- Models that are too simple, that don't even do well on the training data, are said to underfit and also not likely to generalize well



### K-Nearest Neighbors: Classification & Regression

Classification: Given a training set X_Train withb labels y_train, and given a new instance x_test to be classified:

1. Find the most similar instances (X_NN) to x_test that are in X_train
2. Get the labels for y_NN for the instances in X_NN
3. Predict the label for x_test by combining the labels y_NN (e.g. majority vote)



Regression: The R^2 (R-squared) regression score:
- Measures how well a prediction model for regression fits the given data
- The score is between 0 and 1
	- A value of 0 corresponds to a constant prediction of the mean value of all the training targets
	- A value of 1 corresponds to a perfect prediction


Model Complexity:
 - `n_neighbors`: - number of nearest neighbors (k)

Model Ftting:
- `metric`: distance function between points (note: default euclidean setting works well for most datasets)


### Linear Models

#### Linear Regression

- Definition: a linear model is a sum of weighted variables that predicts a target output value given an input instance (e.g. predicting housing prices)
- input instance - feature vector: x = {x0, x1, ... , xn}
- Predicted Output - y = w0x0 + w1x1 + ... + wnxn + b
- Parameters to estimate:
	- w(hat) = (w0, ..., wn)
	- b(hat) = bias term or intercept

Definition: Ordinary Least Squares
- Finds the w and b that minimizes the mean squared error of the model (i.e. the sum of the squared differences b/w target and actual values)
- No parameters to control model complexity (except # of features)
- Finds the values of w and b that minimizes the sum of squared differences (RSS or residual sum of squares) b/w predicted and actual values (ALA mean square error)

The learning algorithm finds the parameters (w, b) that optimize an objective function, typically to minimize some kind of loss function of the predicted values vs actual values.

Note: underscores indicate values that were learned from training data (not set by the user)

Code Snippet
```
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
linreg.intercept_
linreg.coef_
```


#### Ridge Regression

- Ridge regression learns w, b using the same least-squares criterion but adds a penalty for large variations in w parameters
- The addition of a parameter penalty is called "regularization". Regularization prevents overfitting by restricting the model, typically to reduce complexity.
- Ridge regression uses L2 Regularization: minimize sum of squyares of w entries
- The influence of the regularization term is controlled by the alpha parameter. Higher alpha means more regularization and simplier models.

Code Snippet
```
from sklearn.linear_model import Ridge
linridge = Ridge(alpha=20.0).fit(X_train, y_train)
```


#### Feature Normalization

- Important for some ML models that are features are on the same scale (e.g. regularized regression, k-NN, SVMs, NNs)
- One example of feature normalization includes MinMax:
	- For each feature xi: compute the min value xi(min) anbd the max value xi(max) acheived across all instances in the training set
	- For each feature: transform a given feature xi value to a scaled version using the xi' formula: x' = (xi - xi(min)) / (xi(max)-xi(min))
		- essentially, put everything for values between 0 and 1

- Rules for Feature Normalization: The test set must use idebntical scaling to the training set
	- Fit the scaler using the training set, then apply the same scaler to transofrm the test set
	- Do not scale the training set and test sets using different scalers: this could lead to random skew in data
	- Do not fit the scaler using any part of the test set

Code Snippet
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
x_train_scaled = scaler.fit_transform(X_train)
```


#### Lasso Regression

- Lasso regression is another form of regylarized linbear regression that uses an L1 regularization penalty for training (instead of L2 for Ridge)
- L1 Penalty: minimize the sum of the absolute values of the coefficient
- This has the effect of setting parameter weights in w to zero for the least influential variables. This is called a sparse solution, a kind of feature selection.
- Trade off between L1 and L2 Regularization:
	- L2/Ridge: Better for many small or medium sized effects
	- L1/Lasso: Only a few variables with medium/large effects


#### Polynomial Features w/ Linear Regression

- Generate new features consiting of all polynomial combinations of the original two featues (x0, x1)
- The degree of the polynomial specifies how many variables participate at a time in each new feature
- This is still a weighted linear combination of features, so it is still a linear model and can uses the same OLS approach

- Captures interactions between the original features by adding them as features to the linear model
- More generally, we can apply other non-linear transformations to create new features
- Beware of polynomial feature expansion with high degree, as this can lead to complex models that overfit

```
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)
```


#### Logistic Regression

- Y(hat) = logistic (b +w1x1 + ... wn xn )= 1 / 1 + exp[-(b +w1x1 + ... wn xn )]
- Predictions are always between 0 and 1
- The logistic function transforms real-valued input to an output number y between 0 and 1. This output can be interpreted as the probability the input object belongs to the positive class given the input features.
- Parameters for Logistic Regresstion
	- L2 regularization is on by default (like ridge regression)
	- Parmaeter C controls the amount of regularization (default is 1)
	- It can be important to normalize all features so they're on the same scale


```
from sklearn.linear_model import LogisticRegression
LogisticRegression(C=100).fit(X_train, y_train)
```

### Support Vector Machines

- f(x,w,b) = sign (w * x + b) = sign sum (w[i]x[i]+b)
- Dot Product: W * X = (w1, w2) * (x1, x2) = w1x1 + w2x2

- Classifier Margin - Defined as the maximum width the decision boundary area can be increased before hitting a data point.
- Maximum Classifier Margin - The linear classifier with the maximum margin is a linear support vector machine (LSVM)
- The strength of the regularization is determined by C:
	- Larger values of C: less regularization
	- Smaller values of C: more regularization


```
from sklearn.svm import SVC
this_C = 1.0
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
```

```
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
clf = LinearSVC().fit(X_train, y_train)
```


#### Multi-Class Classification

- For Multi-class, Sci-Kit learn automatically detects multiple classes. It then treats this as N number of binary classification problems, and produces binary classifiers for each class.
- 

#### Kernalized Support Vector Machines

- Extension of Linear Support Vector Machines (LSVMs).
- Kernalized Support Vector Machines (SVMs) go beyond the linear case, and can work for both regression and classificatiion.
- Kernalized take original input space and transform it to a higher dimensional feature space, where it becomes easier to classify the transformed data using a linear classifier.
	- Example: transform the data by adding a 2nd dimension/feature. vi = (xi, xi^2)


Definition: a kernal is a similarity measure (modified dot product) between data points


Definition: Radio Basis Function (RBF) Kernal
- K (x, x') = exp [ - y * || x - x'||^2]


Model Complexity:
- Kernal: type of kernal function to be used
- Kernal Parameters:
	- gamma: RBF kernal width
- C: Regularization parameter



Polynomial Kernal Code Snippet

```
from sklearn.svm import SVC
SVC(kernel = 'poly', degree = 3).fit(X_train, y_train)
```

### Cross Validation

Code Snippet for Cross-Validation
```
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors = 5)
cv_scores = cross_val_score(clf, X, y)
```


Validation Curves show sensitivity to changes in an important parameter

```
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(SVC(), X, y,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)
```

### Decision Trees


Decision Tree Splits - Rules to split data into 2 groups.

Informativeness of Splits - an informative split is one that does a good job of splitting the classes. There are a number of mathematical ways to compute the effectiveness (e.g. information gain).

Parameters for Decision Trees
	- Pre-Pruning: Prevent the tree from becoming overly complex
		- `max_depth` - controls maximium depth (number of splits) in the tree
		- `max_leaf_nodes` - max # of leaf nodes
		- `min_samples_leaf` - minimum # of instances that is in a node before splitting further
	- Post-Pruning: Prune back the tree after it has been formed. This is not supported in scikitlearn


```
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
clf = DecisionTreeClassifier().fit(X_train, y_train)
```

Feature Importance
- Can be used to determine which of the features are most relevant to classifying the data

```clf.feature_importances_
```


