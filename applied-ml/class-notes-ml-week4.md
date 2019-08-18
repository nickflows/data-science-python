# Week 4 - Supervised Machine Learning, Part II


## Naive Bayes Classifiers

   - Naive Bayes classifiers are called, Naive, because they make the assumption that each feature instance is independent of all of the others
		- However, features are often correlated
	- Highly efficient at learning and prediction
	- Generalization may not be the best than other classifiers
	- For high dimensional datasets, Naive Bayes can do well to more sophisticated methods (e.g. Support Vector Machines)


### Naive Bayes Classifier Types

- Bernoulli: Binary features
	- Multinomial: Discrete features (e.g. word counts)
	- Gaussian: Continuous, real-valued features
		- Computes mean and standard deviation for each feature
		- Estimates probability that each classes gaussian distribtuon was most likely to generate the data point
		- Usually a parabolic curve across 2 classes

	* Bernoulli and Multinomial are partically well suited for word classifiers.

	- Useful for baseline models compared against more sophisticated models
		- Probability estimates for Naive Bayes aren't always the best


Code Snippet

```
from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB().fit(X_train, y_train)
```

### Random Forest

   - Ensemblle take multiple individual learning models and combines them. 
		- These are effective because each individual model may overfit to a different part of the data. By combining models, we can average out errors.
		- Random Forests are an example of ensembel method applied to decision trees
   - Random Forest - Generation Process
   		1. Original Dataset
   			- `n_estimator` - number of trrees to build
   		2. Randomized Bootstrap copies
   			- Pick N rows at random w/ replacement from the original dataset
   			- Repeat M times 
   		3. Randomized feature splits
   			- When picking the best split for the node, choose from a random subset of features (not the entire feature set)
   			- This is controlled from the max_features parameter (Random Forests are very sensitive to this parameter)
   				- When `max features` is set to 1, leads forests to more complex scenarios
   				- If `max_features` is high, the trees will be similar
   		4. Ensemble prediction
   			- Make a prediction for each tree in the forest
   			- Combine individual predictions
   				- Regression: mean of individual tree predictions
   				- Classification:
   					- Prediction is based on a weighted vote:
   						- Each tree gives a probability for each class
   						- Probabilities averaged across classes
   						- Predict the class with the highest probability

   	- Pros
   		- Widely used, excellent prediction performance on many problems
   		- Doesn't require careful feature normalization
   		- Easily parallelized
   	- Cons
   		- The resulting models are often difficult for humans to interpret
   		- May not be good for high dimensional tasks (e.g. text data, sparse features)


Code Snippet
```
sklearn.ensemble
RandomForestClassifier
RandomForestRegressor
```

`n_estimators` - number of trees to use in ensemble
`max_features` - has a strong effect on performance; influence on diversity of trees
`max_depth` - controls the depth of each tree
`n_jobs` - How many cores to use in parallel training
`random_state` - will produce the same results


### Gradient Boosted Decision Trees (GBDT)

 - Build a series of trees, so that each tree tries to correct the mistakes from the previous trees in the series
 - Build a lot of shallow trees (weak models),
 - The Learning rate controls how hard each tree tries to correct mistakes
 	- A high learning rate leads to more complex trees
 	- A low learning rate leads to simple trees


Code Snippet
```
from sklearn.ensemble import GradientBoostingClassifier
GradientBoostingClassifier().fit(X, y)
```

   	- Pros
   		- Some of the best off-the-shelf accuracy
   		- Inference requires modest memory and is fast
   		- Doesn't require normalization of features, and handles a mixture of features well
   	- Cons
   		- The resulting models are often difficult for humans to interpret
   		- Requires careful tuning of parameters
   		- Training is computational expensive
   		- Doesn't do well with high-dimensional, sparse data

`n_estimators` - number of trees to use in ensemble
`learning_rate` - controls emphasis on fixing errors
`max_depth` - controls the depth of each tree




### Neural Networks

#### Multi-Layer Perceptron
   - Feed Forward Network
   	- Set of input features X = {x0, x1, x2,..., xn}
   	- Hidden Layer: h = {h0, h1, h2}
   	- Output Layer: y
   	- Weight Vectors: 
   		- W, V



   	- Activation Function
   		- tanh activation function (hyperbolic tangent): Maps negative values close to -1, and large positive values close to 1.
   		- relu activation function: Default activation function for Neural Networks. Maps negative input values to 0.
   		- logistic activation function:  
   		- Addition and combination of non-linear activation functions allows neural networks to learn more complicated functions than logistic regression (linear transformation)

   	- Hidden Layers
   		- Hidden Units compute non-linear function of the weighted sums of the input features


Code Snippet
```
from sklearn.neural_network import MLPClassifier
for units, axis in zip([1, 10, 100], subaxes):
    nnclf = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs',
                         random_state = 0).fit(X_train, y_train)
```

2 Layer Multi-layer Perceptron
```
nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs',
                     random_state = 0).fit(X_train, y_train)
```


Multi-layer Perceptron w/ Regularization (`alpha` parameter)
```
    nnclf = MLPClassifier(solver='lbfgs', activation = 'tanh',
                         alpha = this_alpha,
                         hidden_layer_sizes = [100, 100],
                         random_state = 0).fit(X_train, y_train)
```   

`solver` - parmater for algorithm to find the optimal weights. 
