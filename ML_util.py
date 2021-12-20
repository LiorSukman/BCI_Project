import time
import numpy as np
import scipy.io as io
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def asses_model(clf, test, y_test):
    preds = clf.predict(test)  # get predictions according to the trained model
    cor, a_cor, b_cor, c_cor = [0] * 4  # initialize all counters with 0
    class_a_total = np.count_nonzero(y_test == 1)  # Left
    class_b_total = np.count_nonzero(y_test == 2)  # Right
    class_c_total = np.count_nonzero(y_test == 3)  # Idle
    for p, y in zip(preds, y_test):
        if p == y:
            cor += 1
        if y == 1:  # class A (left)
            a_cor += 1 if p == y else 0
        elif y == 2:  # class B (right)
            b_cor += 1 if p == y else 0
        else:  # class C (idle)
            c_cor += 1 if p == y else 0

    print(f"General accuracy on test set is {100 * cor / len(y_test)}%")
    print(f"Accuracy for LEFT class is {100 * a_cor / class_a_total}%")
    print(f"Accuracy for RIGHT class is {100 * b_cor / class_b_total}%")
    print(f"Accuracy for IDLE class is {100 * c_cor / class_c_total}%")


def load_dataset(path):
    print('Loading data...\n')
    X_train = io.loadmat(path + 'FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    y_train = io.loadmat(path + 'LabelTrain.mat')['LabelTrain'].flatten()
    X_test = io.loadmat(path + 'FeaturesTest.mat')['FeaturesTest']
    y_test = io.loadmat(path + 'LabelTest.mat')['LabelTest'].flatten()

    return X_train, y_train, X_test, y_test

def run_gs(model, parameters, X_train, y_train, nfold, seed):
    # Define grid search object with the possible parameters and an N-fold (5) stratified cross validation
    gs = GridSearchCV(model, parameters, cv=StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed), verbose=0)
    print('Starting grid search...\n')
    start = time.time()
    clf = gs.fit(X_train, y_train)  # Actually running the GS
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start))
    print(clf.best_params_)

    return clf
