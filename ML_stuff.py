import numpy as np
import scipy.io as io
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


PATH = "../Sub3/sub-P001/ses-S002/eeg/"

def asses_model(clf, train, test, y_train, y_test):

    preds = clf.predict(test)
    cor = 0
    for p, y in zip(preds, y_test):
        if p == y:
            cor += 1
    print(f"Accuracy on test set is {100*cor/n_test}%")

    preds = clf.predict(train)

    cor = 0
    for p, y in zip(preds, y_train):
        if p == y:
            cor += 1
    print(f"Accuracy on train set is {100 * cor / n_train}%")

if __name__ == "__main__":
    X_train = io.loadmat(PATH + 'FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    y_train = io.loadmat(PATH + 'LabelTrain.mat')['LabelTrain'].T
    X_test = io.loadmat(PATH + 'FeaturesTest.mat')['FeaturesTest']
    y_test = io.loadmat(PATH + 'LabelTest.mat')['LabelTest'].T

    n_train = len(X_train)
    n_test = len(X_test)

    # Non-linear RF classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)
    asses_model(clf, X_train, X_test, y_train, y_test)

    # Linear SVM classifier
    clf = svm.SVC(kernel='linear', random_state=0)
    clf.fit(X_train, y_train)
    asses_model(clf, X_train, X_test, y_train, y_test)




