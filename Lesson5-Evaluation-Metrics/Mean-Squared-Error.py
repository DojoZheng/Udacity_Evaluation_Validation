import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
features = X
labels = y
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=0)

reg1 = DecisionTreeRegressor()
reg1.fit(features_train, labels_train)
dt_mse = mse(labels_test, reg1.predict(features_test))
print "Decision Tree mean squared error: {:.2f}".format(dt_mse)

reg2 = LinearRegression()
reg2.fit(features_train, labels_train)
linear_mse = mse(labels_test, reg2.predict(features_test))
print "Linear regression mean squared error: {:.2f}".format(linear_mse)

results = {
 "Linear Regression": linear_mse,
 "Decision Tree": dt_mse
}