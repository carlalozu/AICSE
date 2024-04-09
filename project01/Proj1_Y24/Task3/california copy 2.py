
# <center>
# <img src="https://habrastorage.org/files/fd4/502/43d/fd450243dd604b81b9713213a247aa20.jpg" />
#
# ## [mlcourse.ai](http://mlcourse.ai) – Open Machine Learning Course
# ### <center> Author: Ilya Larchenko, ODS Slack ilya_l
#
# ## <center> Individual data analysis project


# ## 1. Data description


# __This project is done as a part of  an [open Machine Learning course](http://mlcourse.ai) by [OpenDataScience](http://ods.ai/)__
#
# __I will analyse California Housing Data (1990). It can be downloaded from Kaggle [ https://www.kaggle.com/harrywang/housing ]__


# We will predict the median price of household in block.


from scipy.stats import randint, normaltest
from sklearn.model_selection import (GridSearchCV, learning_curve, validation_curve, KFold, cross_val_score, train_test_split)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from statsmodels.graphics.gofplots import qqplot
from lightgbm.sklearn import LGBMRegressor

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib import pyplot

import pandas as pd
import numpy as np
import os

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


# change this if needed
PATH_TO_DATA = 'data'
full_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'housing.csv'))

X = pd.read_csv(os.path.join(PATH_TO_DATA, 'housing.csv'))


# ## 10. Plotting training and validation curves


# Let's plot Validation Curve

Cs = np.logspace(-5, 4, 10)
train_scores, valid_scores = validation_curve(model, X, y, param_name="alpha",
                                              param_range=Cs, cv=kf, scoring='neg_mean_squared_error')



# We can see that curves for train and CV are very close to each other, it is a sign of underfiting. The difference between the curves does not change along with change in alpha this mean that we should try more complex models comparing to linear regression or add more new features (f.e. polynomial ones)
#
# Using this curve we can find the optimal value of alpha. It is alpha=1. But actually our prediction does not change when alpha goes below 1.
#
# Let's use alpha=1 and plot the learning curve





# Learning curves indicate high bias of the model - this means we will not improve our model by adding more data, but we can try to use more complex models or add more features to improve the results.
#
# This result is inline with the validation curve results. So let's move on to the more complex models.


# ### Random forest


# Actually we can just put all our features into the model but we can easily improve computational performance of the tree-based models, by deleting all monotonous derivatives of features because they does not help at all.
#
# For example, adding log(feature) don't help tree-based model, it will just make it more computationally intensive.
#
# So let's train random forest classifier based on shorten set of the features


features_for_trees = ['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'age_clipped',
                      'longitude', 'latitude', 'housing_median_age', 'total_rooms',
                      'total_bedrooms', 'population', 'households', 'median_income',
                      'distance_to_SF', 'distance_to_LA', 'bedroom/rooms']


X_trees = X[features_for_trees]

model_rf = RandomForestRegressor(n_estimators=100, random_state=17)
cv_scores = cross_val_score(
    model_rf, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

print(np.sqrt(-cv_scores.mean()))


# We can see significant improvement, comparing to the linear model and higher n_estimator probably will help. But first, let's try to tune other hyperparametres:


param_grid = {'n_estimators': [100],
              'max_depth':  [22, 23, 24, 25],
              'max_features': [5, 6, 7, 8]}

gs = GridSearchCV(model_rf, param_grid,
                  scoring='neg_mean_squared_error', n_jobs=-1, cv=kf, verbose=1)

gs.fit(X_trees, y)


print(np.sqrt(-gs.best_score_))


best_depth = gs.best_params_['max_depth']
best_features = gs.best_params_['max_features']


model_rf = RandomForestRegressor(
    n_estimators=100, max_depth=best_depth, max_features=best_features, random_state=17)
cv_scores = cross_val_score(
    model_rf, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

print(np.sqrt(-cv_scores.mean()))


# With the relatively small effort we have got a significant improvement of results. Random Forest results can be further improved by higher n_estimators, let's find the n_estimators at witch the results stabilize.


model_rf = RandomForestRegressor(
    n_estimators=200,  max_depth=best_depth, max_features=best_features, random_state=17)
Cs = list(range(20, 201, 20))
train_scores, valid_scores = validation_curve(
    model_rf, X_trees, y,
    param_name="n_estimators",
    param_range=Cs,
    cv=kf,
    scoring='neg_mean_squared_error'
)

plt.plot(Cs, np.sqrt(-train_scores.mean(axis=1)), 'ro-')

plt.fill_between(x=Cs, y1=np.sqrt(-train_scores.max(axis=1)),
                 y2=np.sqrt(-train_scores.min(axis=1)), alpha=0.1, color="red")


plt.plot(Cs, np.sqrt(-valid_scores.mean(axis=1)), 'bo-')

plt.fill_between(x=Cs, y1=np.sqrt(-valid_scores.max(axis=1)),
                 y2=np.sqrt(-valid_scores.min(axis=1)), alpha=0.1, color="blue")

plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.title('Regularization Parameter Tuning')

plt.show()


# This time we can see that the results of train is much better than CV, but it is totally ok for the Random Forest.
#
# Higher value of n_estimators (>100) does not help much. Let's stick to the n_estimators=200 - it is high enough but not very computationally intensive.


# ### Gradient boosting


# And finally we will try to use LightGBM to solve our problem.
# We will try the model out of the box, and then tune some of its parameters using random search


# uncomment to install if you have not yet
#!pip install lightgbm


model_gb = LGBMRegressor()
cv_scores = cross_val_score(
    model_gb, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=1)

print(np.sqrt(-cv_scores.mean()))


# LGBMRegressor has much more hyperparameters than previous models. As far as this is educational problem we will not spend a lot of time to tuning all of them. In this case RandomizedSearchCV can give us very good result quite fast, much faster than GridSearch. We will do optimization in 2 steps: model complexity optimization and convergence optimization. Let's do it.


# model complexity optimization

param_grid = {'max_depth':  randint(6, 11),
              'num_leaves': randint(7, 127),
              'reg_lambda': np.logspace(-3, 0, 100),
              'random_state': [17]}

gs = RandomizedSearchCV(model_gb, param_grid, n_iter=50, scoring='neg_mean_squared_error',
                        n_jobs=-1, cv=kf, verbose=1, random_state=17)

gs.fit(X_trees, y)


# Let's fix n_estimators=500, it is big enough but is not to computationally intensive yet, and find the best value of the learning_rate


# model convergency optimization

param_grid = {'n_estimators': [500],
              'learning_rate': np.logspace(-4, 0, 100),
              'max_depth':  [10],
              'num_leaves': [72],
              'reg_lambda': [0.0010722672220103231],
              'random_state': [17]}

gs = RandomizedSearchCV(model_gb, param_grid, n_iter=20, scoring='neg_mean_squared_error',
                        n_jobs=-1, cv=kf, verbose=1, random_state=17)

gs.fit(X_trees, y)


# We have got the best params for the gradient boosting and will use them for the final prediction.


# ## 11. Prediction for test or hold-out samples


# Lets sum up the results of our project. We will compute RMSE on cross validation and holdout set and compare them.


results_df = pd.DataFrame(columns=['model', 'CV_results', 'holdout_results'])


# hold-out features and target
X_ho = pd.concat([test_df[dummies_names+['age_clipped']], X_test_scaled,
                 new_features_test_df[new_features_list]], axis=1).reset_index(drop=True)
y_ho = test_df['median_house_value'].reset_index(drop=True)

X_trees_ho = X_ho[features_for_trees]


# linear model
model = Ridge(alpha=1.0)

cv_scores = cross_val_score(
    model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
score_cv = np.sqrt(-np.mean(cv_scores.mean()))


prediction_ho = model.fit(X, y).predict(X_ho)
score_ho = np.sqrt(mean_squared_error(y_ho, prediction_ho))

results_df.loc[results_df.shape[0]] = [
    'Linear Regression',  score_cv,  score_ho]


# Random Forest
model_rf = RandomForestRegressor(
    n_estimators=200,  max_depth=23, max_features=5, random_state=17)

cv_scores = cross_val_score(
    model_rf, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
score_cv = np.sqrt(-np.mean(cv_scores.mean()))


prediction_ho = model_rf.fit(X_trees, y).predict(X_trees_ho)
score_ho = np.sqrt(mean_squared_error(y_ho, prediction_ho))

results_df.loc[results_df.shape[0]] = ['Random Forest',  score_cv,  score_ho]


# Gradient boosting
model_gb = LGBMRegressor(reg_lambda=0.0010722672220103231, max_depth=10,
                         n_estimators=500, num_leaves=72, random_state=17, learning_rate=0.06734150657750829)
cv_scores = cross_val_score(
    model_gb, X_trees, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
score_cv = np.sqrt(-np.mean(cv_scores.mean()))

prediction_ho = model_gb.fit(X_trees, y).predict(X_trees_ho)
score_ho = np.sqrt(mean_squared_error(y_ho, prediction_ho))

results_df.loc[results_df.shape[0]] = [
    'Gradient boosting',  score_cv,  score_ho]


# It seems we have done quite a good job. Cross validation results are inline with holdout ones. Our best CV model - gradient boosting, turned out to be the best on hold-out dataset as well (and it is also faster than random forest).


# ## 12. Conclusions


# To sum up, we have got the solution that can predict the mean house value in the block with RMSE \$46k using our best model - LGB. It is not an extremely precise prediction: \$46k is about 20% of the average mean house price, but it seems that it is near the possible solution for these classes of model based on this data (it is popular dataset but I have not find any solution with significantly better results).
#
# We have used old Californian data from 1990 so it is not useful right now. But the same approach can be used to predict modern house prices (if applied to the resent market data).


# We have done a lot but the results surely can be improved, at least one could try:
#
# - feature engineering: polynomial features, better distances to cities (not Euclidean ones, ellipse representation of cities), average values of target for the geographically closest neighbours (requires custom estimator function for correct cross validation)
# - PCA for dimensionality reduction (I have mentioned it but didn't used)
# - other models (at least KNN and SVM can be tried based on data)
# - more time and effort can be spent on RF and LGB parameters tuning
