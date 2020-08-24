from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# from xgboost import XGBClassifier as MyClassifier
from sklearn.ensemble import RandomForestClassifier as MyClassifier
from lightgbm import LGBMClassifier as MyClassifier
from sklearn.model_selection import GridSearchCV as myCV
# from sklearn.model_selection import RandomizedSearchCV as myCV


def test_accuracy(clf, X_test, y_test):
    output = clf.predict(X_test)
    correct = 0
    for label, out in zip(y_test, output):
        if label == out:
            correct += 1
    correct /= len(X_test)
    return correct

def submit(clf, filename):
    df = pd.read_csv('nmlo-contest-1/test.csv', sep=',',header=None)
    all_data = df.values
    all_data = all_data[1:].astype(np.float32)
    X = all_data[:,1:]
    pred = clf.predict(X)
    labeled = np.zeros((len(pred), 2))
    for i in range(len(pred)):
        labeled[i][0] = all_data[i][0]
        labeled[i][1] = pred[i]
    labeled = labeled.astype(int)
    np.savetxt(filename, labeled, fmt='%i', delimiter=",")


def create():
    estimator = MyClassifier()
    # optimization_dict = {'max_depth': [2,5,8,10,12,15],
    #                     'n_estimators': [100,200,500,1000,2000]
    #                     # ,'learning_rate': [0.001, 0.01, 0.05]
    #                     }

    # optimization_dict = {
    #     'n_estimators': [100, 200, 400, 700, 1000, 2000],
    #     'max_depth': [10,15,20,25],
    #     'max_leaf_nodes': [50, 100, 200]
    # }

    optimization_dict = {
        'n_estimators': [200, 400, 700, 1000],
        'colsample_bytree': [0.7, 0.8],
        'max_depth': [4,5,10,15,20,25],
        'num_leaves': [50, 100, 200],
        'reg_alpha': [1.1, 1.2, 1.3],
        'reg_lambda': [1.1, 1.2, 1.3],
        'min_split_gain': [0.3, 0.4],
        'subsample': [0.7, 0.8, 0.9],
        'subsample_freq': [20]
    }

    model = myCV(estimator, optimization_dict, n_jobs=-1, cv=10,
                        scoring='accuracy', verbose=1)

    print("Using:", type(estimator), type(model))

    model.fit(X,y)
    print(model.best_score_)
    print(model.best_params_)


    accuracy = test_accuracy(model, X_test, y_test)
    print("Accuracy:", accuracy)
    return model, accuracy
#    submit(clf, "submission.csv")


df = pd.read_csv('nmlo-contest-1/train.csv', sep=',',header=None)
all_data = df.values
all_data = all_data[1:].astype(np.float32)
X, y = all_data[:,:11], all_data[:,11]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.1, random_state=15)


# max_rand = 60

# best = 0
# bestClf = None
# for depth in range(2, 20):
#     avgForDepth = 0
#     for rand in range(0, max_rand):
#         clf, accuracy = create(depth, rand)
#         if accuracy > best:
#             best = accuracy
#             bestClf = clf
#         print("Depth: {}, Rand: {}, Accuracy: {}".format(depth, rand, accuracy))
#         avgForDepth += accuracy
#     avgForDepth /= max_rand
#     print("Average for depth {} is {}".format(depth, avgForDepth))

# print("Best accuracy: {}".format(best))
clf, accuracy = create()
print("Accuracy:", accuracy)

submit(clf, "submission3.csv")
