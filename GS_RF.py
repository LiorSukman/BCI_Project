import numpy as np
from sklearn.ensemble import RandomForestClassifier

import ML_util

PATH = "./data/"
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


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = ML_util.load_dataset(PATH)

    # Define parameters to check in the grid search, Note that we use logarithmic scales!
    n_estimatorss = np.logspace(n_estimators_min, n_estimators_max, n_estimators_num).astype('int')
    max_depths = np.logspace(max_depth_min, max_depth_max, max_depth_num).astype('int')
    min_samples_splits = np.logspace(min_samples_splits_min, min_samples_splits_max, min_samples_splits_num,
                                     base=2).astype('int')
    min_samples_leafs = np.logspace(min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, base=2).astype(
        'int')

    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths, 'min_samples_split': min_samples_splits,
                  'min_samples_leaf': min_samples_leafs}  # organize options in a dictionary

    # Initialize RF model with constant parameters (seed for reproducability and class weight for unbalanced datasets)
    model = RandomForestClassifier(random_state=SEED, class_weight='balanced')

    clf = ML_util.run_gs(model, parameters, X_train, y_train, NFOLD, SEED)

    ML_util.asses_model(clf, X_test, y_test)  # evaluate the model

