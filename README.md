They are my firs machine learning scripts, after reading some books, and scikit-learn tutorial

I'm trying to explain my path of thinking, so it's a lot of comments
Also I haven't amended script, after I saw a mistake, so it is some bad decisions, and miss conceptions in it

The repo contain:

1 urea_stone:
Classification, with knn, and svc models
result is rater mediocre, but perhaps it is issue of data

2 diabetes:
Classification with Naive Bayes, and decision tree
Result is excellent with 100% accuracy in train, and test data, but one time more data issue
Also some feature selection mistakes have been made

3 cancer_risk:
Prediction with multinomial Naive Bayes, and RandomForest
Good results in prediction, some mistakes in thinking, about good model to choose

4 perceptron
My implementation of perceptron classifier, base on another implementation in book
In that project I reimplement perceptron written with numpy to polars dataframes

5 comparisons
Actually comparison of numpy perceptron with polars perceptron execution time
In future perhaps some more efficiency benchmarks

6 adaline
My implementation of adaptive linear neuron (adaline), base on another implementation in book
In that project I reimplement adaline written with numpy to polars dataframes

7 sun_spots
Time series forecasting, with scikit learn SGDRegressor. Moderate good results R^2 > 0.85 on both sets
Also I develop my features selecting abilities, and use L1 penalty to identify irrelevant features during training