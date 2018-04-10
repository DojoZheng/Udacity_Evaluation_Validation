# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic-data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB


# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
features = X
labels = y
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=0)

clf1 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
decision_tree_recall = recall(labels_test, clf1.predict(features_test))
decision_tree_precision = precision(labels_test, clf1.predict(features_test))
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(decision_tree_recall, decision_tree_precision)

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
nb_recall = recall(labels_test, clf2.predict(features_test))
nb_precision = precision(labels_test, clf2.predict(features_test))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(nb_recall, nb_precision)
# clf2.fit(X, y)
# print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(y,clf2.predict(X)),precision(y,clf2.predict(X)))

results = {
  "Naive Bayes Recall": nb_recall,
  "Naive Bayes Precision": nb_precision,
  "Decision Tree Recall": decision_tree_recall,
  "Decision Tree Precision": decision_tree_precision
}