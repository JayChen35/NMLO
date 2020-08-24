from dataparser import get_training_data, split, get_testing_data, submit
# from xgboost import XGBRegressor as myRegressor
from sklearn.ensemble import RandomForestRegressor as MyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV as myCV
import pickle
import numpy as np


X, y = get_training_data()
X_train, y_train, X_val, y_val = split(X, y, 0.2)
print(X_train.shape, y_train.shape)

def create():
    estimator = MyRegressor()
    # optimization_dict = {'max_depth': [2,5,8,10,12,15],
    #                     'n_estimators': [100,200,500,1000,2000]
    #                     # ,'learning_rate': [0.001, 0.01, 0.05]
    #                     }

    optimization_dict = {
        'n_estimators': [100, 200, 400, 800, 1000, 1500, 2000],
        'max_features': ['auto'],
        'max_depth': [None] + [10, 20, 40, 60, 100, 120],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True]
    }

    optimization_dict = {
        'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        'max_features': ['auto'],
        'max_depth': [None] + [int(x) for x in np.linspace(10, 110, num = 11)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True]
    }


    # optimization_dict = {
    #     'n_estimators': [200, 400, 700, 1000],
    #     'colsample_bytree': [0.7, 0.8],
    #     'max_depth': [4,5,10,15,20,25],
    #     'num_leaves': [50, 100, 200],
    #     'reg_alpha': [1.1, 1.2, 1.3],
    #     'reg_lambda': [1.1, 1.2, 1.3],
    #     'min_split_gain': [0.3, 0.4],
    #     'subsample': [0.7, 0.8, 0.9],
    #     'subsample_freq': [20]
    # }

    print(optimization_dict)

    model = myCV(estimator, optimization_dict, n_jobs=-1, n_iter=250, cv=10,
                        scoring='neg_mean_absolute_error', verbose=1)

    print("Using:", type(estimator), type(model))

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_params_)

    # predictions = model.predict(X_val)
    # MAE = mean_absolute_error(y_val , predictions)
    # print("MAE:", MAE)
    return model


# clf = create()
data = {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}
# data = {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True}
clf = MyRegressor(**data)

# clf.fit(X_train, y_train)
# predictions = clf.predict(X_val)
# MAE = mean_absolute_error(y_val , predictions)
# print("MAE:", MAE)

clf.fit(X, y)
test = get_testing_data()
predictions = clf.predict(test)
submit(predictions, "submission3.csv")

# submit(clf, "submission.csv")
