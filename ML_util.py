import time
import numpy as np
import scipy.io as io
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_pipeline(path, pca_n=10, per=0.25, seed=0):
    print('Loading data...\n')
    features = io.loadmat(path + 'Features.mat')#['FeaturesTrainSelected']
    labels = io.loadmat(path + 'Labels.mat')#['FeaturesTrainSelected']

    print(f'Splitting data (test fraction={per}; seed={seed})...\n')
    x_train, y_train, x_test, y_test = split_data(features, labels, per, seed)

    print('Scaling data...\n')
    x_train, x_test, scaler = scale_data(x_train, x_test)

    print(f'Applying PCA on data (n components={pca_n})...\n')
    x_train, x_test, pca = apply_pca(x_train, x_test, n=pca_n)

    return x_train, y_train, x_test, y_test, scaler, pca


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler.transform(X_train), scaler.transform(X_test), scaler


def apply_pca(X_train, X_test, n):
    pca = PCA(n_components=n, whiten=True)
    pca.fit(X_train)

    return pca.transform(X_train), pca.transform(X_test), pca


def split_data(features, labels, per=0.25, seed=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=per, random_state=seed)
    train_index, test_index = sss.split(features, labels).__next__()
    train, train_labels = features[train_index], labels[train_index]
    test, test_labels = features[test_index], labels[test_index]

    return train, train_labels, test, test_labels

    """np.random.seed(seed)
    inds = np.arange(len(features))
    inds.shuffle()
    features, labels = features[inds], labels[inds]
    left, right, idle = labels == 1, labels == 2, labels == 3
    nleft, nright, nidle = np.count_nonzero(left), np.count_nonzero(right), np.count_nonzero(idle)

    left_labels, right_labels, idle_labels = labels[left], labels[right], labels[idle]
    left_fets, right_fets, idle_fets = features[left], features[right], features[idle]

    test = np.concatenate((left_fets[:per * nleft], right_fets[:per * nright], idle_fets[:per * nidle]), axis=0)
    test_labels = np.concatenate((left_labels[:per * nleft], right_labels[:per * nright], idle_labels[:per * nidle]),
                                 axis=0)
    test_inds = np.arange(len(test))
    test_inds.shuffle()

    train = np.concatenate((left_fets[per * nleft:], right_fets[per * nright:], idle_fets[per * nidle:]), axis=0)
    train_labels = np.concatenate((left_labels[per * nleft:], right_labels[per * nright:], idle_labels[per * nidle:]),
                                  axis=0)
    train_inds = np.arange(len(train))
    train_inds.shuffle()

    return train[train_inds], train_labels[train_inds], test[test_inds], test_labels[test_inds]"""


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
    X_train = io.loadmat(path + 'FeaturesTrainSelected.mat')['FeaturesTrainSelected']
    y_train = io.loadmat(path + 'LabelTrain.mat')['LabelTrain'].flatten()
    X_test = io.loadmat(path + 'FeaturesTest.mat')['FeaturesTest']
    y_test = io.loadmat(path + 'LabelTest.mat')['LabelTest'].flatten()

    return X_train, y_train, X_test, y_test


def run_gs(model, parameters, X_train, y_train, nfold, seed):
    # Define grid search object with the possible parameters and an N-fold (5) stratified cross validation
    gs = GridSearchCV(model, parameters, cv=StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed), verbose=0)
    print('Starting grid search...')
    start = time.time()
    clf = gs.fit(X_train, y_train)  # Actually running the GS
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start))

    return clf


def plot_cf(clf, X_test, y_test):
    display_labels = ['Left', 'Right', 'Idle']
    labels = [1, 2, 3]
    predictions = clf.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_test, predictions, labels=labels)
    disply = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disply.plot()
    plt.show()
    return disply.ax_
