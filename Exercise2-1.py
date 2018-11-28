import os
import tarfile
from six.moves import urllib
import numpy as np

# download housing dataset
DOWNLOAD_ROOT = "http://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()

#create a test set
#stratified smapling
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5) #to have discrete categories
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)
from sklearn.model_selection import  StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2, random_state= 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis = 1, inplace = True)

    housing = strat_train_set.drop("median_house_value", axis = 1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis = 1) #drop text attribute
    #define dataframeselector
    from sklearn.base import BaseEstimator, TransformerMixin
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[self.attribute_names].values

    # combine attributes
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no*args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y = None):
            return self # nothing else to do
        def transform(self, X, y = None):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    from sklearn.base import TransformerMixin
    class MyLabelBinarizer(TransformerMixin):
        def __init__(self, *args, **kwargs):
            self.encoder = LabelBinarizer(*args, **kwargs)

        def fit(self, X, y=0):
            self.encoder.fit(X)
            return self

        def transform(self, X, y=0):
            return self.encoder.transform(X)

    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelBinarizer


    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy = 'median')),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])
    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('label_binarizer', MyLabelBinarizer()),
        ])
    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ])
    housing_prepared = full_pipeline.fit_transform(housing)

    from sklearn.svm import SVR
    svm_poly_reg = SVR(kernel= "poly", degree= 3, C = 1000, epsilon= 0.1)
    svm_poly_reg.fit(housing_prepared, housing_labels)

    # some_data = housing.iloc[:5]
    # some_labels = housing_labels.iloc[:5]
    # some_data_prepared = full_pipeline.transform(some_data)
    # print("Predictions:\t", svm_poly_reg.predict(some_data_prepared))
    # print("Labels:\t\t", list(some_labels))

    #measure rmse
    from sklearn.metrics import mean_squared_error
    housing_predictions = svm_poly_reg.predict(housing_prepared)
    svr_mse = mean_squared_error(housing_labels, housing_predictions)
    svr_rmse = np.sqrt(svr_mse)
    print(svr_rmse)



