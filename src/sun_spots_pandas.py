import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf


data = pd.read_fwf('./data/sunspot.txt', header=None, names=['years', 'sun_spots'])

# For feature selection, in pacf, we can see for lag 8 is last relevant correlation
plot_pacf(data['sun_spots'])
plt.show()

feature_number = 8
for lag in range(1, feature_number + 1):
    data[f'sun_spots_lag{lag}'] = data['sun_spots'].shift(lag)

data = data.dropna()

features = data.drop(['years', 'sun_spots'], axis=1)
target = data['sun_spots']

train_features, test_features, train_target, test_target = (
    train_test_split(features, target, shuffle=False, train_size=0.8))

# use strong l1 regularization to see features are irrelevant
neuron = SGDRegressor(penalty='l1', eta0=0.00005, alpha=10, random_state=42)
neuron.fit(train_features, train_target)
train_score = neuron.score(test_features, test_target)

print('_____________________ results for 8 features _____________________')
print('Train score:', train_score)
print('Weights:', neuron.coef_)

# Select relevant features with not 0 weights
train_features = train_features.iloc[:, [0, 1, 2, 6, 7]]
test_features = test_features.iloc[:, [0, 1, 2, 6, 7]]

# retrain model with smaller penalty
neuron.alpha = 0.1
neuron.fit(train_features, train_target)
train_score = neuron.score(test_features, test_target)
neuron.fit(test_features, test_target)
test_score = neuron.score(test_features, test_target)

print()
print('____________________ results for best features ____________________')
print('Train score:', train_score)
print('Test score:', test_score)

# Plot chart, to better show prediction quality
prediction = neuron.predict(test_features)
years = data['years'].tail(len(prediction))
plt.plot(years, test_target, 'b-', label='real data')
plt.plot(years, prediction, 'r-', label='prediction')
plt.xlabel('years')
plt.ylabel('sun spots')
plt.legend()
plt.show()
