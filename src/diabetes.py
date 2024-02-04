from math import ceil, floor

import polars as pl
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


data = pl.read_csv('./data/diabetes.csv')

# Data exploration

# Change format for more useful to computation, integers instead strings
data = data.select(
    pl.col('Age'),
    pl.when(pl.col('Gender') == 'Male').then(1).when(pl.col('Gender') == 'Female').then(2).name.keep(),
    pl.when(cs.string().exclude('Gender', 'class') == 'Yes').then(1)
    .when(cs.string().exclude('Gender', 'class') == 'No').then(0).name.keep(),
    pl.when(pl.col('class') == 'Positive').then(1).when(pl.col('class') == 'Negative').then(0).name.keep()
)

correlation = data.corr()
print(correlation)
# There is many columns with various correlation to target, from very week to moderately strong

# Features selection

# Drop columns with weak, and very week correlation to target
target_correlation = correlation.get_column('class')
columns_names = data.columns
target_correlation = pl.DataFrame({'column_name': columns_names, 'correlation_to_target': target_correlation})
relevant_correlation = target_correlation.filter(pl.col('correlation_to_target').abs() > 0.4)
relevant_correlation = relevant_correlation.filter(pl.col('column_name') != 'class')
relevant_correlation = relevant_correlation.sort('correlation_to_target')

relevant_columns = relevant_correlation.get_column('column_name')
ax = sns.barplot(x=relevant_columns, y=relevant_correlation.get_column('correlation_to_target'))
ax.tick_params(axis='x', labelrotation=30)
ax.set(xlabel='')
plt.show()
data = data.select(*relevant_columns, 'class')

# All used data are actually a binary data, perhaps Bernoulli naive bayes is a good model

rows_number = data.select(pl.len())
rows_number = rows_number.item()
print('Rows number:', rows_number)
positive_numbers = data.select(pl.arg_where(pl.col('class') == 1)).select(pl.len())
positive_numbers = positive_numbers.item()
print('Positive_number:', positive_numbers)
majority_group = round(max(positive_numbers, rows_number - positive_numbers)/rows_number*100, 2)
print('Majority group occurrence:', majority_group)
# The major group is 61.54%


# Sampling data
shuffle_set = data.sample(fraction=1, shuffle=True, seed=42)  # the target values are not distributed random
training_set_size = 0.75
training_set = shuffle_set.head(ceil(rows_number*training_set_size))
testing_set = shuffle_set.tail(floor(rows_number*(1 - training_set_size)))

training_target = training_set.get_column('class')
testing_target = testing_set.get_column('class')

training_features = training_set.select(pl.col(*relevant_columns))
testing_features = testing_set.select(pl.col(*relevant_columns))

bnb = BernoulliNB()
bnb.fit(training_features, training_target)
training_score = bnb.score(training_features, training_target)
print('\n--------------------- BernoulliNB ---------------------')
print('training score:', round(training_score, 2))
testing_score = bnb.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# 86% for training and 85% for test, not bad for first model.


dtc = DecisionTreeClassifier()
dtc.fit(training_features, training_target)
training_score = dtc.score(training_features, training_target)
print('\n--------------------- decision tree ---------------------')
print('training score:', round(training_score, 2))
testing_score = dtc.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Look some better, 90% for training and 88% for test, model don't look very overfit,
# what is easily possible with than just decision tree

# Try to use decision tree with all features, and overfit it

training_features = training_set
testing_features = testing_set


dtc.fit(training_features, training_target)
training_score = dtc.score(training_features, training_target)
print('\n--------------------- decision tree all features ---------------------')
print('training score:', round(training_score, 2))
testing_score = dtc.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Ok it was some unexpected for me 100% on train, and 100% on test.
# The feature selecting was too restrictive, and I loose relevant information

# These data set is seem to easy to predict everything correct,
# It is still place for improvement, some features are certainly unnecessary,
# Also it is probably possible to choose model with less computation resources need
# However I want to try some automated machine learning with harder to predict output data
