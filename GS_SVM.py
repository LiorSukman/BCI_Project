import numpy as np
from sklearn import svm

import ML_util

PATH = "./data/"
NFOLD = 5
SEED = 0

min_gamma = -8
max_gamma = 0
num_gamma = 9
min_c = 0
max_c = 6
num_c = 7


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = ML_util.load_dataset(PATH)

    # Define parameters to check in the grid search, Note that we use logarithmic scales!
    gammas = np.logspace(min_gamma, max_gamma, num_gamma)
    cs = np.logspace(min_c, max_c, num_c)

    parameters = {'C': cs, 'gamma': gammas}

    # Initialize RF model with constant parameters (seed for reproducability and class weight for unbalanced datasets)
    model = svm.SVC(kernel='linear', random_state=SEED, class_weight='balanced')

    clf = ML_util.run_gs(model, parameters, X_train, y_train, NFOLD, SEED)

    ML_util.asses_model(clf, X_test, y_test)  # evaluate the model

