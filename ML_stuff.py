import numpy as np
import scipy.io as io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import time
from sklearn import svm

PATH = "../Sub3/sub-P001/ses-S002/eeg/"
NFOLD = 5
SEED = 0

n_estimators_min = 0
n_estimators_max = 2
n_estimators_num = 3
max_depth_min = 1
max_depth_max = 2
max_depth_num = 2
min_samples_splits_min = 1
min_samples_splits_max = 5
min_samples_splits_num = 5
min_samples_leafs_min = 0
min_samples_leafs_max = 5
min_samples_leafs_num = 6

def asses_model(clf, test, y_test):
    preds = clf.predict(test)
    cor, a_cor, b_cor, c_cor = [0] * 4
    class_a_total = np.count_nonzero(y_test == 0)
    class_b_total = np.count_nonzero(y_test == 1)
    class_c_total = np.count_nonzero(y_test == 2)
    for p, y in zip(preds, y_test):
        if p == y:
            cor += 1
        if y == 0:
            a_cor += 1 if p == y else 0
        elif y == 1:
            b_cor += 1 if p == y else 0
        else:
            c_cor += 1 if p == y else 0

    print(f"General accuracy on test set is {100 * cor / len(y_test)}%")
    print(f"Accuracy for class A is {100 * a_cor / class_a_total}%")
    print(f"Accuracy for class B is {100 * b_cor / class_b_total}%")
    print(f"Accuracy for class C is {100 * c_cor / class_c_total}%")


if __name__ == "__main__":
    X_train = io.loadmat(PATH + 'FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    y_train = io.loadmat(PATH + 'LabelTrain.mat')['LabelTrain'].T
    X_test = io.loadmat(PATH + 'FeaturesTest.mat')['FeaturesTest']
    y_test = io.loadmat(PATH + 'LabelTest.mat')['LabelTest'].T

    n_train = len(X_train)
    n_test = len(X_test)

    n_estimatorss = np.logspace(n_estimators_min, n_estimators_max, n_estimators_num).astype('int')
    max_depths = np.logspace(max_depth_min, max_depth_max, max_depth_num).astype('int')
    min_samples_splits = np.logspace(min_samples_splits_min, min_samples_splits_max, min_samples_splits_num,
                                     base=2).astype('int')
    min_samples_leafs = np.logspace(min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, base=2).astype(
        'int')

    print()
    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths, 'min_samples_split': min_samples_splits,
                  'min_samples_leaf': min_samples_leafs}
    model = RandomForestClassifier(random_state=SEED, class_weight='balanced')
    gs = GridSearchCV(model, parameters, cv=StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED), verbose=0)
    print('Starting grid search...')
    start = time.time()
    clf = gs.fit(features, labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start))
    print(clf.best_params_)

    n_estimators = gs.best_params_['n_estimators']
    max_depth = gs.best_params_['max_depth']
    min_samples_split = gs.best_params_['min_samples_split']
    min_samples_leaf = gs.best_params_['min_samples_leaf']
    print('Best hyperparameters are:', gs.best_params_)

    asses_model(clf, test, y_test)

