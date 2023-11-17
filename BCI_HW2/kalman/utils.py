import numpy as np


def split_data(X_kf,y_kf,data, training_range, valid_range, testing_range):

    # Number of examples after taking into account bins removed for lag alignment
    num_examples_kf = X_kf.shape[0]

    # Note that each range has a buffer of 1 bin at the beginning and end
    # This makes it so that the different sets don't include overlapping data
    training_set = np.arange(np.int(np.round(training_range[0] * num_examples_kf)) + 1,
                             np.int(np.round(training_range[1] * num_examples_kf)) - 1)
    testing_set = np.arange(np.int(np.round(testing_range[0] * num_examples_kf)) + 1,
                            np.int(np.round(testing_range[1] * num_examples_kf)) - 1)
    valid_set = np.arange(np.int(np.round(valid_range[0] * num_examples_kf)) + 1,
                          np.int(np.round(valid_range[1] * num_examples_kf)) - 1)

    # Get training data
    X_kf_train = X_kf[training_set, :]
    y_kf_train = y_kf[training_set, :]

    # Get testing data
    X_kf_test = X_kf[testing_set, :]
    y_kf_test = y_kf[testing_set, :]

    # Get validation data
    X_kf_valid = X_kf[valid_set, :]
    y_kf_valid = y_kf[valid_set, :]
    # %%
    # Z-score inputs
    X_kf_train_mean = np.nanmean(X_kf_train, axis=0)
    X_kf_train_std = np.nanstd(X_kf_train, axis=0)
    X_kf_train = (X_kf_train - X_kf_train_mean) / X_kf_train_std
    X_kf_test = (X_kf_test - X_kf_train_mean) / X_kf_train_std
    X_kf_valid = (X_kf_valid - X_kf_train_mean) / X_kf_train_std

    # Zero-center outputs
    y_kf_train_mean = np.mean(y_kf_train, axis=0)
    y_kf_train = y_kf_train - y_kf_train_mean
    y_kf_test = y_kf_test - y_kf_train_mean
    y_kf_valid = y_kf_valid - y_kf_train_mean
    # %%
    return (X_kf_train, y_kf_train), (X_kf_valid, y_kf_valid), (X_kf_test, y_kf_test), (X_kf_train_mean, y_kf_train_mean)
    pass
