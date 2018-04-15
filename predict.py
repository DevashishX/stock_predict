from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class regressor(Ridge):
    pass


def scale_data(X_train, X_test):
    """Scales data using Minmaxscaler in range -3 to 3
    returns scaled train and test data"""
    cols = X_train.shape()[1]
    X_train_scaled = np.array()
    X_test_scaled = np.array()
    scaler = MinMaxScaler(feature_range=(-3, 3), copy=True)

    for i in range(0, cols):
        scaler.fit(X_train[:, i])
        X_train_scaled[:, i] = scaler.transform(X_train[:, i])
        X_test_scaled[:, i] = scaler.transform(X_test[:, i])

    return X_train_scaled, X_test_scaled
