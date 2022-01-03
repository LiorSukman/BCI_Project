import time
import numpy as np
import scipy.io as io
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def data_stats(labels, name):
    for label in np.unique(labels):
        print(f"    Number of samples with label {label} in {name} is {np.count_nonzero(labels == label)}")


def load_pipeline(path, pca_n=10, per=0.25, seed=0):
    print('Loading data...\n')
    features = io.loadmat(path + 'Features.mat')  # ['FeaturesTrainSelected']
    labels = io.loadmat(path + 'Labels.mat')  # ['FeaturesTrainSelected']
    data_stats(labels, name='data')

    print(f'Splitting data (test fraction={per}; seed={seed})...\n')
    x_train, y_train, x_test, y_test = split_data(features, labels, per, seed)
    data_stats(y_train, name='training set')
    data_stats(y_test, name='test set')

    print('Scaling data...\n')
    x_train, x_test, scaler = scale_data(x_train, x_test)

    print(f'Applying PCA on data (n components={pca_n})...\n')
    x_train, x_test, pca = apply_pca(x_train, x_test, n=pca_n)

    return x_train, y_train, x_test, y_test, scaler, pca


def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    return scaler.transform(x_train), scaler.transform(x_test), scaler


def apply_pca(x_train, x_test, n):
    pca = PCA(n_components=n, whiten=True)
    pca.fit(x_train)

    return pca.transform(x_train), pca.transform(x_test), pca


def split_data(features, labels, per=0.25, seed=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=per, random_state=seed)
    train_index, test_index = sss.split(features, labels).__next__()
    train, train_labels = features[train_index], labels[train_index]
    test, test_labels = features[test_index], labels[test_index]

    return train, train_labels, test, test_labels


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
    print(f"Accuracy for class A is {100 * a_cor / class_a_total}%")
    print(f"Accuracy for class B is {100 * b_cor / class_b_total}%")
    print(f"Accuracy for class C is {100 * c_cor / class_c_total}%")


def load_dataset(path):
    x_train = io.loadmat(path + 'FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    y_train = io.loadmat(path + 'LabelTrain.mat')['LabelTrain'].flatten()
    x_test = io.loadmat(path + 'FeaturesTest.mat')['FeaturesTest']
    y_test = io.loadmat(path + 'LabelTest.mat')['LabelTest'].flatten()

    return x_train, y_train, x_test, y_test


def run_gs(model, parameters, x_train, y_train, nfold, seed):
    # Define grid search object with the possible parameters and an N-fold (5) stratified cross validation
    gs = GridSearchCV(model, parameters, cv=StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed), verbose=0)
    print('Starting grid search...')
    start = time.time()
    clf = gs.fit(x_train, y_train)  # Actually running the GS
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start))

    return clf


def plot_cf(clf, x_test, y_test):
    display_labels = ['Left', 'Right', 'Idle']
    labels = [1, 2, 3]
    predictions = clf.predict(x_test)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    disply = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disply.plot()
    plt.show()
    return disply.ax_
