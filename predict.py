from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.datasets import make_regression, load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import stk


def scale_data(X_train, X_test, X_today, classtype="StandardScaler"):
    """Scales data using MinMaxScaler in range -3 to 3
    returns scaled train and test data"""

    cols = X_train.shape[1]
    rows_train = X_train.shape[0]
    rows_test = X_test.shape[0]
    rows_today = X_today.shape[0]

    X_train_scaled = np.ones((rows_train, 1))
    X_test_scaled = np.ones((rows_test, 1))
    X_today_scaled = np.ones((rows_today, 1))

    # X_train, X_test = norm_pre(X_train, X_test)

    if(classtype == "MinMax"):
        scaler = MinMaxScaler(feature_range=(-3, 3))
    elif(classtype == "Robust"):
        scaler = RobustScaler()
    if(classtype == "StandardScaler"):
        scaler = StandardScaler()

    for i in range(1, cols):
        scaler.fit(X_train[:, i].reshape(-1, 1))
        X_train_scaled = np.c_[X_train_scaled,
                               scaler.transform(X_train[:, i].reshape(-1, 1))]
        X_test_scaled = np.c_[X_test_scaled,
                              scaler.transform(X_test[:, i].reshape(-1, 1))]
        X_today_scaled = np.c_[X_today_scaled,
                               scaler.transform(X_today[:, i].reshape(-1, 1))]

    return X_train_scaled, X_test_scaled, X_today_scaled


def poly_pre(X, n):
    poly = PolynomialFeatures(n, interaction_only=False)
    poly.fit(X)
    return poly.transform(X)


def PCA_pre(X, n=20):
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.transform(X)


def norm_pre(X_train):
    norm = Normalizer()
    norm.fit(X_train)
    return norm.transform(X_train)


def nonlin_comp(X):
    m = X.shape[1]
    m = int(m / 10)
    shrink = PCA_pre(X, n=m)
    shrink_sin = np.sin(shrink)
    shrink_cos = np.cos(shrink)
    return np.c_[X, shrink_sin, shrink_cos]


def plot(classifier, X, y, title="data"):
    # b: blue g: green r: red c: cyan m: magenta y: yellow k: black w: white
    l1 = plt.plot(y)
    l2 = plt.plot(classifier.predict(X))
    plt.setp(l1, label='Real', color='b', lw=1, ls='-', marker='+', ms=1.5)
    plt.setp(l2, label='Prediction', color='r',
             lw=1, ls='--', marker='o', ms=1.5)
    plt.title(title)
    plt.ylabel("Target")
    plt.xlabel("Sample Number")
    plt.legend()
    plt.show()


def plotchange(Window, classifier, X, y, title="Model Vs Real"):
    # b: blue g: green r: red c: cyan m: magenta y: yellow k: black w: white
    l1 = Window.a1.plot(y, label='Real', color='b',
                        lw=1, ls='-', marker='+', ms=1.5, zorder=3)
    l2 = Window.a1.plot(classifier.predict(X), label='Prediction', color='r',
                        lw=1, ls='--', marker='o', ms=1.5, zorder=3)
    # Window.a1.setp(l1, )
    # Window.a1.setp(l2, )
    Window.a1.set_title(title)
    Window.a1.set_ylabel("Target")
    Window.a1.set_xlabel("Sample/Day Number")
    Window.a1.legend()
    Window.a1.grid()
    # Window.a1.show()


def fit_company_change(name):
    #2337, 2330, 6223, 6220
    X_close, X_index, X_close_index, y_close, y_index, y_close_index = ProperReturn(
        name)
    X = X_close_index
    y = y_close_index
    X = nonlin_comp(X)
    X = poly_pre(X, 4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, stratify=None, test_size=0.25, random_state=42)
    X_train = X_test = X
    X_today = X[-1:, :].reshape(-1, X.shape[1])
    X_train, X_test, X_today = scale_data(
        X_train, X_test, X_today, "StandardScaler")
    retx = X_train
    rety = y
    return retx, rety


def fit_algo(X_train, X_test, y_train, y_test, algotype="Ridge"):
    """fit the given algorithm to given data, returns an object of type classifier"""
    train_list = []
    test_list = []
    print("using:", algotype)

    if(algotype == "Ridge"):
        algotype = Ridge(alpha=0.1, max_iter=20000)

    elif(algotype == "Lasso"):
        algotype = Lasso(alpha=0.1, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print("train score: ", classifier.score(X_train, y_train),
              "test score: ", classifier.score(X_test, y_test))
        return classifier

    elif(algotype == "LinearRegression"):
        algotype = LinearRegression()
        classifier = algotype.fit(X_train, y_train)
        print("train score: ", classifier.score(X_train, y_train),
              "test score: ", classifier.score(X_test, y_test))
        return classifier

    elif(algotype == "Ridge1"):
        algotype = Ridge(alpha=1, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print("train score: ", classifier.score(X_train, y_train),
              "test score: ", classifier.score(X_test, y_test))
        return classifier

    elif(algotype == "Ridge0.1"):
        algotype = Ridge(alpha=0.1, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print("train score: ", classifier.score(X_train, y_train),
              "test score: ", classifier.score(X_test, y_test))
        return classifier

    elif(algotype == "Ridge0.01"):
        algotype = Ridge(alpha=0.01, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print("train score: ", classifier.score(X_train, y_train),
              "test score: ", classifier.score(X_test, y_test))
        return classifier

    elif(algotype == "RidgeCV"):
        cv_array = [float(float(i) / 100.0) for i in range(-100, 1000)]
        cv_array[cv_array.index(0)] = 0.1
        algotype = RidgeCV(cv_array)
        classifier = algotype.fit(X_train, y_train)
        print("train score: ", classifier.score(X_train, y_train),
              "test score: ", classifier.score(X_test, y_test))
        return classifier

    cv_array = [float(float(i) / 100.0) for i in range(-1000, 1000)]
    print(max(cv_array), min(cv_array))
    for i in cv_array:
        algotype = Ridge(alpha=i, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        train_list.append(train_score)
        test_list.append(test_score)

    optimal_score = max(train_list)
    optimal_score_index = train_list.index(optimal_score)
    final_alpha = cv_array[train_list.index(optimal_score)]
    algotype = Ridge(alpha=final_alpha, max_iter=20000)
    print("alpha:", final_alpha, optimal_score, test_list[optimal_score_index])

    return classifier


def fit_neural_network(X_train, X_test, y_train, y_test, activation="relu",
                       network_structure=(10, 10), learn_rate=0.001, iter=20000):
    NN = MLPRegressor(hidden_layer_sizes=network_structure,
                      learning_rate_init=learn_rate, max_iter=iter, )
    classifier = NN.fit(X_train, y_train)
    print("The score for train set is: {}".format(
        classifier.score(X_train, y_train)))
    print("The score for test set is: {}".format(
        classifier.score(X_test, y_test)))
    return classifier


def ProperReturn(name):
    close, index, close_index = stk.csv_to_df(name)
    close = np.array(close, dtype=np.float64)
    index = np.array(index, dtype=np.float64)
    close_index = np.array(close_index, dtype=np.float64)
    y_close = close[1:, :]
    y_index = index[1:, :]
    close_index = close_index[1:, :]
    collength = y_close.shape[0]
    X_close = X_index = np.arange(0, collength).reshape(-1, 1)
    y_close_index = close_index[:, 0].reshape(-1, 1)
    X_close_index = close_index[:, 1].reshape(-1, 1)
    X_close_index = np.c_[X_close_index,
                          np.arange(0, collength).reshape(-1, 1)]

    return X_close, X_index, X_close_index, y_close, y_index, y_close_index


def fit_company(name):
    #2337, 2330, 6223, 6220
    X_close, X_index, X_close_index, y_close, y_index, y_close_index = ProperReturn(
        name)
    X = X_close_index
    y = y_close_index
    X = nonlin_comp(X)
    X = poly_pre(X, 4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, stratify=None, test_size=0.1, random_state=42)
    X_today = X[-1:, :].reshape(-1, X.shape[1])
    X_train, X_test, X_today = scale_data(
        X_train, X_test, X_today, "StandardScaler")
    # classifier = fit_neural_network(
    #    X_train, X_test, y_train.ravel(), y_test.ravel(), network_structure=(10, 10, 10), activation="relu")
    classifier = fit_algo(X_train, X_test, y_train,
                          y_test, algotype="RidgeCV")
    print("for ", str(name), " Todays's Price is: ",
          str(classifier.predict(X_today)))
    plot(classifier, X_train, y_train, "train plot")
    plot(classifier, X_test, y_test, "test plot")
    return classifier


def predict_all():
    #2337, 2330, 6223, 6220
    company_list = [2337, 2330, 6223, 2867]
    clasifier_list = []
    for i in company_list:
        clasifier_list.append(fit_company(i))
    return clasifier_list


def main():
    #2337, 2330, 6223, 2867
    company_list = [2337, 2330, 6223, 2867]
    clasifier_list = []
    for i in company_list:
        clasifier_list.append(fit_company(i))
    fit_company_change(2337)
    return clasifier_list


if __name__ == "__main__":
    main()
