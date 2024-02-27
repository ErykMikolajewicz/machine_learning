import polars as pl
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

# getting data from file with fixed-width formatted lines, probably much easier in pandas with read_fwf
data = pl.read_csv('./data/sunspot.txt', has_header=False, new_columns=['data'])
data = data[:-1]  # to drop last row what is null
data = data.select(pl.col('data').str.strip_chars().str.split(' ').alias('split_data'))
data = data.select(pl.col("split_data").list.to_struct()).unnest("split_data")
data = data.drop('field_1')
data = data.rename({'field_0': 'year', 'field_2': 'sun_spots'})
data = data.select(pl.col('*').cast(pl.Float64))

plot_pacf(data.select('sun_spots'))
plt.show()

"""Looks like 2 previous years are most important for forecast
Also 8-th previous number of sun spots can be relevant, others are probably unimportant
I will try use SGDRegressor with data contain last 8 years as a features, also I will use moderately strong 
L1 regularization, it should change irrelevant features coefficients near to 0
"""

feature_number = 8
lagged_data = data.select(pl.col('*').slice(feature_number))
lagged_data = lagged_data.with_columns(data.select(
    pl.col('sun_spots')
    .alias(f'sun_spots_lag{feature_number - i}')
    .filter(pl.col('year') >= pl.col('year').first() + i)
    .slice(0, len(lagged_data))
    for i in range(feature_number))
)

features = lagged_data.select(pl.exclude('year', 'sun_spots'))
target = lagged_data.select(pl.col('sun_spots'))

train_features, test_features, train_target, test_target = train_test_split(features,
                                                                            target,
                                                                            shuffle=False,
                                                                            train_size=0.8)

neuron = SGDRegressor(penalty='l1', eta0=0.000005, max_iter=100_000, alpha=2)
neuron.fit(train_features, train_target)
train_score = neuron.score(test_features, test_target)
neuron.fit(test_features, test_target)
test_score = neuron.score(test_features, test_target)

print('_____________________ results for 8 features _____________________')
print('Train score:', train_score)
print('Test score:', test_score)
print('Weights:', neuron.coef_)
"""The score is R^2 value, so everything greater than 0 is valuable, result for train set 0.85 what is moderately good,
for test is even better with 0.87. Weights for features with lags: 1, 2, 3 and 8 seem relevant, but feature with lag2
is surprisingly small, lesser than for lag3 and lag8, if we talk about absolute values. For alpha = 2 some features 
drop to 0, so selecting features via regularization work."""


# It's easier to assess is r^2 > 0.85 great, so small visualization will be useful
prediction = neuron.predict(test_features)
years = lagged_data.select(pl.col('year').tail(len(prediction)))
plt.plot(years, test_target, 'b-', label='real data')
plt.plot(years, prediction, 'r-', label='prediction')
plt.xlabel('years')
plt.ylabel('sun spots')
plt.legend()
plt.show()
"""Looks ok, both lines on chart are quite similar. Probably I could done more, check model with only relevant features,
or previously only initially train model, then drop irrelevant features, and end model training, also it's possible, to
check other learning rates, regularization parameters, etc. but I consider this prediction, as mostly done."""
