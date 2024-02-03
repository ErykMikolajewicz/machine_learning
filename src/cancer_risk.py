import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


data = pl.read_csv('./data/cancer.csv')

# Some data preparation
data: pl.DataFrame = data.drop('Patient Id', 'index')
data = data.select(pl.all().exclude('Level'),
                   pl.when(pl.col('Level') == 'Low').then(1).when(pl.col('Level') == 'Medium').then(2).
                   when(pl.col('Level') == 'High').then(3).name.keep()
                   )

correlation = data.corr()
columns_names = data.columns
axes = sns.heatmap(correlation, xticklabels=columns_names, yticklabels=columns_names, annot=True, cbar=False)
axes.xaxis.tick_top()
plt.show()
# There is a lot of data, usually with good correlation to target
# I decide to drop all features with correlation less than 0.6
# In dataset number 2 it was a bad decision
# But here is more features, 12 has correlation grater than 0.6
# Using more features can easily overfit model, and make harder to analyse distribution, and collinearity
# Analyse it is important, because I want to try is linear logistic accurate model for that data
# Unfortunately data for linear logistic should be normally distributed, and not collinear

target_correlation = correlation.get_column('Level')
target_correlation = pl.DataFrame({'column_name': columns_names, 'correlation_to_target': target_correlation})
relevant_correlation = (target_correlation.filter(pl.col('correlation_to_target').abs() > 0.6)
                        .filter(pl.col('column_name') != 'Level').sort('correlation_to_target'))


relevant_col_names = relevant_correlation.get_column('column_name')
correlation = data.select(*relevant_col_names).corr()
plt.clf()
axes = sns.heatmap(correlation, xticklabels=relevant_col_names, yticklabels=relevant_col_names, annot=True, cbar=False)
axes.xaxis.tick_top()
plt.show()
# I just realized, I don't have range data, so Logistic Regression is inappropriate model
# Looking for collinearity probably don't have sens.
# Again try model from bayes family, but I will check MultinomialNB,
# because there is more options in features, that only true/false

# Check majority group to have reference point is model good
major_counts = data.group_by('Level').len().max().select('len')
all_counts = data.get_column('Level').len()
majority_group = major_counts/all_counts * 100
# Majority group is 36,5%

features = data.select(pl.exclude('Level'))
target = data.select('Level')
train_features, test_features, train_target, test_target = train_test_split(features,
                                                                            target,
                                                                            stratify=target,
                                                                            random_state=42,
                                                                            train_size=0.75)
model_train_features = train_features.select(*relevant_col_names)
model_test_features = test_features.select(*relevant_col_names)

mnb = MultinomialNB()

mnb.fit(model_train_features, train_target)
training_score = mnb.score(model_train_features, train_target)
print('\n--------------------- MultinomialNB relevant features ---------------------')
print('train score:', round(training_score, 2))
testing_score = mnb.score(model_test_features, test_target)
print('test score:', round(testing_score, 2))
# 66% for train set, and 58% for test. Better than majority group, but not very well

# Check result for same features, despite age, what is not categorical data
model_train_features = train_features.select(pl.exclude('Age', 'Level'))
model_test_features = test_features.select(pl.exclude('Age', 'Level'))

mnb.fit(model_train_features, train_target)
training_score = mnb.score(model_train_features, train_target)
print('\n--------------------- MultinomialNB most features ---------------------')
print('train score:', round(training_score, 2))
testing_score = mnb.score(model_test_features, test_target)
print('test score:', round(testing_score, 2))
# Okay, one time more, better result, 80% for train model, and 84% for test, perhaps more features usually mean better


