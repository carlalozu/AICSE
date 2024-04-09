
"""Data preprocessing for the California housing dataset following the provided reference"""
import os
import warnings

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore') # `do not disturbe` mode


# change this if needed
PATH_TO_DATA = 'data'
full_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'housing.csv'))


train_df, test_df = train_test_split(
    full_df, shuffle=True, test_size=0.25, random_state=17)
train_df = train_df.copy()
test_df = test_df.copy()


numerical_features = list(train_df.columns)
numerical_features.remove('ocean_proximity')
numerical_features.remove('median_house_value')
print(numerical_features)


train_df['median_house_value_log'] = np.log1p(train_df['median_house_value'])
test_df['median_house_value_log'] = np.log1p(test_df['median_house_value'])


skewed_features = ['households', 'median_income',
                   'population', 'total_bedrooms', 'total_rooms']
log_numerical_features = []
for f in skewed_features:
    train_df[f + '_log'] = np.log1p(train_df[f])
    test_df[f + '_log'] = np.log1p(test_df[f])
    log_numerical_features.append(f + '_log')


max_house_age = train_df['housing_median_age'].max()


train_df['age_clipped'] = train_df['housing_median_age'] == max_house_age
test_df['age_clipped'] = test_df['housing_median_age'] == max_house_age


lin = LinearRegression()

# we will train our model based on all numerical non-target features with not NaN total_bedrooms
appropriate_columns = train_df.drop(['median_house_value', 'median_house_value_log',
                                     'ocean_proximity', 'total_bedrooms_log'], axis=1)
train_data = appropriate_columns[~pd.isnull(train_df).any(axis=1)]

# model will be validated on 25% of train dataset
# theoretically we can use even our test_df dataset (as we don't use target) for
# this task, but we will not
temp_train, temp_valid = train_test_split(
    train_data, shuffle=True, test_size=0.25, random_state=17)


# Obviously our linear regression approach is much better. Let's train our model
# on whole train dataset
# and apply it to the rows with blanks. But preliminary we will "remember" the
# rows with NaNs, because there is a chance, that it can contain useful information.


lin.fit(train_data.drop(['total_bedrooms'], axis=1),
        train_data['total_bedrooms'])

train_df['total_bedrooms_is_nan'] = pd.isnull(train_df).any(axis=1).astype(int)
test_df['total_bedrooms_is_nan'] = pd.isnull(test_df).any(axis=1).astype(int)


dropped_cols = ['median_house_value', 'median_house_value_log', 'total_bedrooms',
                'total_bedrooms_log', 'ocean_proximity', 'total_bedrooms_is_nan']

train_df['total_bedrooms'].loc[pd.isnull(train_df).any(axis=1)] =\
    lin.predict(train_df.drop(dropped_cols,
                              axis=1)[pd.isnull(train_df).any(axis=1)])

test_df['total_bedrooms'].loc[pd.isnull(test_df).any(axis=1)] =\
    lin.predict(test_df.drop(dropped_cols,
                             axis=1)[pd.isnull(test_df).any(axis=1)])

# linear regression can lead to negative predictions, let's change it
test_df['total_bedrooms'] = test_df['total_bedrooms'].apply(
    lambda x: max(x, 0))
train_df['total_bedrooms'] = train_df['total_bedrooms'].apply(
    lambda x: max(x, 0))


# Let's update 'total_bedrooms_log' and check if there are no NaNs left


train_df['total_bedrooms_log'] = np.log1p(train_df['total_bedrooms'])
test_df['total_bedrooms_log'] = np.log1p(test_df['total_bedrooms'])


# After filling of blanks let's have a closer look on dependences between some numerical features


# It seems there are no new insights about numerical features (only confirmation of the old ones).
#
# Let's try to do the same thing but for the local (geographically) subset of our data.


# the point near which we want to look at our variables
local_coord = [-122, 41]
euc_dist_th = 2  # distance treshhold

euclid_distance = train_df[['latitude', 'longitude']].apply(
    lambda x: np.sqrt((x['longitude']-local_coord[0])**2 +
                      (x['latitude']-local_coord[1])**2), axis=1)

# indicate wethere the point is within treshhold or not
indicator = pd.Series(euclid_distance <= euc_dist_th, name='indicator')

print("Data points within treshhold:", sum(indicator))


ocean_proximity_dummies = pd.get_dummies(
    pd.concat([train_df['ocean_proximity'], test_df['ocean_proximity']]),
    drop_first=True)


dummies_names = list(ocean_proximity_dummies.columns)


train_df = pd.concat(
    [train_df, ocean_proximity_dummies[:train_df.shape[0]]], axis=1)
test_df = pd.concat(
    [test_df, ocean_proximity_dummies[train_df.shape[0]:]], axis=1)

train_df = train_df.drop(['ocean_proximity'], axis=1)
test_df = test_df.drop(['ocean_proximity'], axis=1)


sf_coord = [-122.4194, 37.7749]
la_coord = [-118.2437, 34.0522]

train_df['distance_to_SF'] = np.sqrt(
    (train_df['longitude']-sf_coord[0])**2+(train_df['latitude']-sf_coord[1])**2)
test_df['distance_to_SF'] = np.sqrt(
    (test_df['longitude']-sf_coord[0])**2+(test_df['latitude']-sf_coord[1])**2)

train_df['distance_to_LA'] = np.sqrt(
    (train_df['longitude']-la_coord[0])**2+(train_df['latitude']-la_coord[1])**2)
test_df['distance_to_LA'] = np.sqrt(
    (test_df['longitude']-la_coord[0])**2+(test_df['latitude']-la_coord[1])**2)


features_to_scale = numerical_features + \
    log_numerical_features+['distance_to_SF', 'distance_to_LA']

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(train_df[features_to_scale]),
                              columns=features_to_scale, index=train_df.index)
X_test_scaled = pd.DataFrame(scaler.transform(test_df[features_to_scale]),
                             columns=features_to_scale, index=test_df.index)


kf = KFold(n_splits=10, random_state=17, shuffle=True)


# ### Linear regression


# For the first initial baseline we will take Ridge model with only initial
# numerical and OHE features


model = Ridge(alpha=1)
X = train_df[numerical_features+dummies_names]
y = train_df['median_house_value']
cv_scores = cross_val_score(
    model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(-cv_scores.mean()))


# We are doing cross validation with 10 folds, computing 'neg_mean_squared_error'
#(neg - because sklearn needs scoring functions to be minimized).
# Our final metrics: RMSE=np.sqrt(-neg_MSE)


# Visually is not obvious so let's try to create a couple of new variables and check:


new_features_train_df = pd.DataFrame(index=train_df.index)
new_features_test_df = pd.DataFrame(index=test_df.index)


new_features_train_df['1/distance_to_SF'] = 1 / \
    (train_df['distance_to_SF']+0.001)
new_features_train_df['1/distance_to_LA'] = 1 / \
    (train_df['distance_to_LA']+0.001)
new_features_train_df['log_distance_to_SF'] = np.log1p(
    train_df['distance_to_SF'])
new_features_train_df['log_distance_to_LA'] = np.log1p(
    train_df['distance_to_LA'])

new_features_test_df['1/distance_to_SF'] = 1/(test_df['distance_to_SF']+0.001)
new_features_test_df['1/distance_to_LA'] = 1/(test_df['distance_to_LA']+0.001)
new_features_test_df['log_distance_to_SF'] = np.log1p(
    test_df['distance_to_SF'])
new_features_test_df['log_distance_to_LA'] = np.log1p(
    test_df['distance_to_LA'])


# We can also generate some features correlated to the prosperity:
# - rooms/person - how many rooms are there per person. The higher - the richer people
# are living there - the more expensive houses they buy
# - rooms/household - how many rooms are there per family. The similar one but
# corresponds to the number of rooms per family (assuming household~family), not per person.
# - two similar features but counting only bedrooms


new_features_train_df['rooms/person'] = train_df['total_rooms'] / \
    train_df['population']
new_features_train_df['rooms/household'] = train_df['total_rooms'] / \
    train_df['households']

new_features_test_df['rooms/person'] = test_df['total_rooms'] / \
    test_df['population']
new_features_test_df['rooms/household'] = test_df['total_rooms'] / \
    test_df['households']


new_features_train_df['bedrooms/person'] = train_df['total_bedrooms'] / \
    train_df['population']
new_features_train_df['bedrooms/household'] = train_df['total_bedrooms'] / \
    train_df['households']

new_features_test_df['bedrooms/person'] = test_df['total_bedrooms'] / \
    test_df['population']
new_features_test_df['bedrooms/household'] = test_df['total_bedrooms'] / \
    test_df['households']


# - the luxurity of house can be characterized buy number of bedrooms per rooms


new_features_train_df['bedroom/rooms'] = train_df['total_bedrooms'] / \
    train_df['total_rooms']
new_features_test_df['bedroom/rooms'] = test_df['total_bedrooms'] / \
    test_df['total_rooms']


# - the average number of persons in one household can be the signal of prosperity or
#  the same time the signal of richness but in any case it can be a useful feature


new_features_train_df['average_size_of_household'] = train_df['population'] / \
    train_df['households']
new_features_test_df['average_size_of_household'] = test_df['population'] / \
    test_df['households']


# And finally let's scale all this features


new_features_train_df = pd.DataFrame(
    scaler.fit_transform(new_features_train_df),
    columns=new_features_train_df.columns,
    index=new_features_train_df.index)

new_features_test_df = pd.DataFrame(
    scaler.transform(new_features_test_df),
    columns=new_features_test_df.columns,
    index=new_features_test_df.index)


# We will add new features one by one and keeps only those that improve our best score


# computing current best score

X = pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled],
              axis=1, ignore_index=True)

cv_scores = cross_val_score(
    model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
best_score = np.sqrt(-cv_scores.mean())
print("Best score: ", best_score)

# list of the new good features
new_features_list = []

for feature in new_features_train_df.columns:
    new_features_list.append(feature)
    X = pd.concat([train_df[dummies_names+['age_clipped']], X_train_scaled,
                   new_features_train_df[new_features_list]
                   ],
                  axis=1, ignore_index=True)
    cv_scores = cross_val_score(
        model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    score = np.sqrt(-cv_scores.mean())
    if score >= best_score:
        new_features_list.remove(feature)
        print(feature, ' is not a good feature')
    else:
        print(feature, ' is a good feature')
        print('New best score: ', score)
        best_score = score


# We have got 5 new good features. Let's update our X variable


final_ds = pd.concat([train_df[dummies_names+['age_clipped', 'median_house_value']],
               X_train_scaled,
               new_features_train_df[new_features_list]
               ],
              axis=1).reset_index(drop=True)

# Save the final dataset
final_ds.to_csv(os.path.join(PATH_TO_DATA, 'final_ds.csv'), index=False)
