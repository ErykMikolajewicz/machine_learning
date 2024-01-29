from math import ceil, floor

import polars as pl
import seaborn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

data = pl.read_csv('./data/kidney_stone.csv')

# Data exploration

describe = data.describe()
print(describe)
# There is no null values to fill.
# Also, no values with high cardinality to immediately drop

# Normalization std data
mean = describe.filter(pl.col('statistic') == 'mean').drop('statistic')
std = describe.filter(pl.col('statistic') == 'std').drop('statistic')
coefficient_of_variation = std/mean
print('Coefficients of variation:')
print(coefficient_of_variation)
# gravity have got very small variation, it is probably necessary to drop
# also variation for ph is small

# Making plot for Pearson correlation
correlation = data.corr()
columns_names = correlation.schema.keys()
axes = seaborn.heatmap(correlation, xticklabels=columns_names, yticklabels=columns_names, annot=True, cbar=False)
axes.xaxis.tick_top()
fig = axes.get_figure()
fig.savefig('./plots/urea_stone/correlation_plot')
# pH and cond have very week relation to target, they are necessary to drop
# Another problem is high collinearity between some values, especially:
# gravity - urea - osmo
# cond - osmo

rows_number = data.select(pl.len())
rows_number = rows_number.item()
print('Rows number:', rows_number)
positive_numbers = data.select(pl.arg_where(pl.col("target") == 1)).select(pl.len())
positive_numbers = positive_numbers.item()
print('Positive_number:', positive_numbers)
majority_group = round(max(positive_numbers, rows_number - positive_numbers)/rows_number*100, 2)
print('Majority group occurrence:', majority_group)
# The major group is 57%, what is minimal efficient for model to have any value

# Looking for data distribution

for column_name in columns_names:
    if column_name == 'target':
        continue
    plt.clf()
    plotting_data = data.get_column(column_name)
    axes = seaborn.histplot(plotting_data)
    fig = axes.get_figure()
    fig.savefig(f'./plots/urea_stone/histograms/{column_name}')
# Most distributions look like normal distribution with skewness
# Calc distribution is very not look as normal distribution.

# Sampling data
shuffle_set = data.sample(fraction=1, shuffle=True, seed=42)  # the target values are not distributed random
training_set_size = 0.6
training_set = shuffle_set.head(ceil(rows_number*training_set_size))
testing_set = shuffle_set.tail(floor(rows_number*(1 - training_set_size)))

training_target = training_set.get_column('target')
testing_target = testing_set.get_column('target')

# Feature selection

# I have decided to drop columns with small correlation to target
# Also it's necessary to solve problem with collinearity
# I have decided to drop column gravity, because of small variation
# osmo, is dropped because weaker relation to target than urea

training_features = training_set.select(pl.col('urea', 'calc'))
testing_features = testing_set.select(pl.col('urea', 'calc'))

# The first model to train will be k nearest neighbors, it works for non gaussian data and is simply

# Plot the scatter plot
plt.clf()
axes = seaborn.scatterplot(x=data.get_column('urea'), y=data.get_column('calc'), hue=data.get_column('target'))
fig = axes.get_figure()
fig.savefig('./plots/urea_stone/urea_calcium_scatterplot')
# The points with same target are apparently closer, some outliers are in groups of other colors
# Also points in boundaries can be hard to classify, but I can see some potential for this data, and algorithm

knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
model = knn.fit(training_features, training_target)
training_score = model.score(training_features, training_target)
print('\n--------------------- knn model: urea, calcium ---------------------')
print('training score:', round(training_score, 2))
testing_score = model.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))

# 61% in test score, rather not a success modifying neighbors number don't help
# Using weight='distance make awesome thing in training data, but completely failed on test data
# Perhaps feature selection was made bad
# Try with gravity despite urea parameter, has better correlation with target

# Feature selection, second approach, gravity instead urea

# Plot another scatter plot
plt.clf()
axes = seaborn.scatterplot(x=data.get_column('gravity'), y=data.get_column('calc'), hue=data.get_column('target'))
fig = axes.get_figure()
fig.savefig('./plots/urea_stone/gravity_calcium_scatterplot')
# Hard to say is it look better, there still are outliers, and hard to classify points on boards

training_features = training_set.select(pl.col('gravity', 'calc'))
testing_features = testing_set.select(pl.col('gravity', 'calc'))

knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=7)
model = knn.fit(training_features, training_target)
training_score = model.score(training_features, training_target)
print('\n--------------------- knn model: gravity, calcium ---------------------')
print('training score:', round(training_score, 2))
testing_score = model.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# For n_neighbors=7 75% for training, and 74% for test, much better, but still not very well

# I will try with features rejected by to small relation to target, perhaps it will make a change

training_features = training_set.select(pl.col('gravity', 'calc', 'ph', 'cond'))
testing_features = testing_set.select(pl.col('gravity', 'calc', 'ph', 'cond'))

knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
model = knn.fit(training_features, training_target)
training_score = model.score(training_features, training_target)
print('\n--------------------- knn model: include features with week correlation ---------------------')
print('training score:', round(training_score, 2))
testing_score = model.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Training set have better accuracy with 85%, but test set is even worst - 71%
# better stay with gravity, and calc only, other params are noise.

# Last try for knn, all features, without any data preparation:

training_features = training_set
testing_features = testing_set

knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
model = knn.fit(training_features, training_target)
training_score = model.score(training_features, training_target)
print('\n--------------------- knn model: all features ---------------------')
print('training score:', round(training_score, 2))
testing_score = model.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Worst result for all, 67% in training, and 58% in test

# Test linear svc model with best output from knn

training_features = training_set.select(pl.col('gravity', 'calc'))
testing_features = testing_set.select(pl.col('gravity', 'calc'))

lsvc = LinearSVC(C=1, dual='auto', max_iter=10_000)
lsvc.fit(training_features, training_target)
training_score = lsvc.score(training_features, training_target)
print('\n--------------------- linear svc model: gravity, calc ---------------------')
print('training score:', round(training_score, 2))
testing_score = lsvc.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Same result, as with knn model and same data 75% and 74%, perhaps it is all, what achievable,
# changing C parameter seem to have no effect

# Test poly nominal svc model

lsvc = SVC(C=1, max_iter=10_000, degree=3, kernel='poly')
lsvc.fit(training_features, training_target)
training_score = lsvc.score(training_features, training_target)
print('\n--------------------- poly svc model: gravity, calc ---------------------')
print('training score:', round(training_score, 2))
testing_score = lsvc.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Slightly better, than with linear 79% for training, and 77% for test

# Try svc with some collinear data removed in feature selection. Add urea data

training_features = training_set.select(pl.col('gravity', 'calc', 'urea'))
testing_features = testing_set.select(pl.col('gravity', 'calc', 'urea'))

lsvc = SVC(C=0.5, max_iter=10_000, degree=3, kernel='poly')
lsvc.fit(training_features, training_target)
training_score = lsvc.score(training_features, training_target)
print('\n--------------------- poly svc model: gravity, calc, urea ---------------------')
print('training score:', round(training_score, 2))
testing_score = lsvc.score(testing_features, testing_target)
print('test score:', round(testing_score, 2))
# Much worst result 62% on training set, and 52% on test set, using urea data was a bad decision


# Conclusions:
# Not using gravity, because of small variation wasn't good idea
# Exclusion functions with high collinearity was good decision,
# however it could by, because their low correlation to target
# KNN model have better prognosis than majority group so is useful, but score is not impressing
# Similarly svc model. SVC with poly nominal was slightly better, but it is probably not worth extra cost
# in production environment, with dealing with more complex model
# Perhaps approximately 75% for this model is all what is achievable
# Training, and testing group are some unbalanced, despite various size have similar number of 1 in target 16 and 18
# Perhaps using larger data set will solve that problem
