## Week 3 - Model Evaluation and Selection


### Evaluation Metrics
	- Choose evaluation metric based on context
	- Compute your evaluation metrics for multiple models

### Accuracy w/ Imbalanced Classes

- Suppose you have 2 classes: Relevant Class (Positive) and Not Relevant (Negative)
- Out of 1000 items, on average, 1 is the positive class
- Accuracy = Correct Predictions / Total Instances



### Dummy Classifier

- Completely ignore the input data
- Can be used as a sanity check on imbalanced datasets for trained classifiers
- They provide a "null metric" baseline 
- Commonly used strategies for Dummy Classifiers:
	- stratified: random predictions based on training set class distribution
	- uniform: generates predicitions uniformly at random
	- most_frequent: predicts the most frequent label in the training set 
	- constant: always predicts a constant label provided by a user

- What does it mean if accuracy is close to dummy baseline?
	- Wrong parameters
	- Poorly formed features
	- Class imbalance

```
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_dummy_predictions = dummy_majority.predict(X_test)
```

### Dummy Regressors
- strategy parameter options
	- mean
	- median
	- quantile
	- constant


### Confusion Matrix

- Each test instance is in exactly one box (MECE)

	Predicted Negative   Predicted Positive
True Negative
True Positive


```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_majority_predicted)
```

### Other Evaluation Metrics

- Accuracy = TP + TN / (TP + TN + FP + FN)
- Classification Error = 1 - accuracy
- Precision = TP / (TP + FP)
	- What % of positive instances are correct?
- Recall (AKA Trye Positive Rate) = TP / (TP + FN)
	- what % of all positive instances did the classifier correctly identify as positive?

- Specificity (False Positive Rate) = FP / (TN + FP)
	- What % of all negative instances does the classifier incorrectly identify as positive?

- F1 = 2 * Precision * Recall / (Precision + Recall) 
	- Combines precision and recall into a single number (harmonic mean)

- F Score - (1+B^2) * (precision * recall ) /(B^2 * precision + recall)
	- B allows for the adjustment of control between precision and recall
	- Precision Goal: set B=0.5 (false positives hurt performance more than false negatives)
	- Recall Goal: set B = 2 (False negatives hurt performanfce more than false positives)


Code Snippet
```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
f1_score(y_test,predictions)

from sklearn.metrics import classification_report
print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
```


### Decision Functions

- Each classifier score value per test point indicates how confidently the classifier predicts the positive class (large positive) or negative class (large negative)
- Choosing a fixed decision threshold gives a classification rule. When this is tuned between values, you get a set of classification outcomes.

- Predicited probability of class membership (predict_proba)
	- Choose most likely class
	- Adjusting threshold affects predictions of a classifier
	- Higher threshold predicts more conservative classifier
	- Not all models provide realistic probability estimates


Code Snippet
```
```

### Precision & Recall Curves

- Shows the trade-off between precision and recall for different decision thresholds
- Optimal point (top right where both values are equal to 1)


#### Precision vs Recall Trade-off
- Increasing the recall of the classifier comes at the cost of precision (and vice versa)
- Recall - optimize for correctly identifying positive examples
- Precision - optimize for not showing incorrect or unhelpful information (e.g. search engine ranking, annotations)


```
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
closest_zero = np.argmin(np.abs(thresholds)) // get values where precision and recall trade-off is set to 0
```


#### ROC Curves


```
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
```


### Multi-Class Evaluation

- Multi-class evaluation is an extension of evaluation for the binary class case
- Overall, evaluation metrics are averages across classes (also, imbalanced classes are important to consider)


```
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,10)], columns = [i for i in range(0,10)])

print(classification_report(y_test_mc, svm_predicted_mc))
```

- Micro-average precision: Each instance has equal weight
	1. Aggregate outcomes across all classes
	2. Largest classes have the most influence (essentially a weighted average by class)

- Macro-average precision: Each class has equal weight	
	1. Compute metrics within each class
	2. Average resulting metrics across average

`parameter: average=micro`

### Regression Evaluation Metrics

	- r2_score -- usually the best for most cases
	- mean_squared_error - squared difference between target and predicted values
	- mean_absolute_error - absolute difference between target and predicted values 
	- median_absolute_error - robust to outliers

### Model Selection using Evaluation Metrics

Three Approaches to evaluating a model
	1. train / test on the same data
	2. Single train/test split
	3. K-Fold Cross Validation

Cross-Validation Code Snippet

```
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc')
```

Grid Search







