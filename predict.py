from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.datasets import make_regression
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class result():
    pass


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

    #X_train, X_test = norm_pre(X_train, X_test)

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
    poly = PolynomialFeatures(n, interaction_only=True)
    poly.fit(X)
    return poly.transform(X)


def PCA_pre(X, n=20):
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.transform(X)


def norm_pre(X_train, X_today=None):
    norm = Normalizer()
    norm.fit(X_train)
    # if(X_today != None):
    # return norm.transform(X_train), norm.transform(X_today)
    # else:
    return norm.transform(X_train)


def fit_algo(X_train, X_test, y_train, y_test, algotype="Ridge"):
    """fit the given algorithm to given data, returns an object or type classifier"""
    train_list = [[], []]
    test_list = [[], []]
    print("using:", algotype)

    if(algotype == "Ridge"):
        algotype = Ridge(alpha=0.1, max_iter=20000)
    elif(algotype == "Lasso"):
        algotype = Lasso(alpha=0.1, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print(classifier.score(X_test, y_test))
        return classifier
    elif(algotype == "LinearRegression"):
        algotype = LinearRegression()
        classifier = algotype.fit(X_train, y_train)
        print(classifier.score(X_test, y_test))
        return classifier
    elif(algotype == "Ridge1"):
        algotype = Ridge(alpha=1, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print(classifier.score(X_test, y_test))
        return classifier
    elif(algotype == "Ridge0.1"):
        algotype = Ridge(alpha=0.1, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print(classifier.score(X_test, y_test))
        return classifier
    elif(algotype == "Ridge0.01"):
        algotype = Ridge(alpha=0.01, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        print(classifier.score(X_test, y_test))
        return classifier

    for i in range(-20, 21):
        alpha_calc = float(float(i) / 10.0)
        algotype = Ridge(alpha=alpha_calc, max_iter=20000)
        classifier = algotype.fit(X_train, y_train)
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        train_list[0].append(train_score)
        train_list[1].append(alpha_calc)
        test_list[0].append(test_score)
        test_list[1].append(alpha_calc)

    optimal_score = max(test_list[0])
    optimal_score_index = test_list[0].index(optimal_score)
    final_alpha = float(test_list[1][optimal_score_index])
    algotype = Ridge(alpha=final_alpha, max_iter=20000)
    print("final alpha:", final_alpha, train_list[0][optimal_score_index],
          test_list[0][optimal_score_index], sep="\t")
    #print(train_list[optimal_score_index], max(test_list), sep="\n")
    # plt.plot(train_list)
    # plt.plot(test_list)
    # plt.show()
    return classifier


def main():
    #samples = int(input("Number of Samples: "))
    #X, y = make_regression(n_samples=samples)
    bunch = load_boston(return_X_y=True)
    X = np.array(bunch[0])
    y = np.array(bunch[1])

    X = poly_pre(X, 3)
    #X = norm_pre(X)
    #X = PCA_pre(X, )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, stratify=None, test_size=0.2, random_state=42)

    X_today = X_train[-1:, :].reshape(-1, X_train.shape[1])

    X_train, X_test, X_today = scale_data(X_train, X_test, X_today, "MinMax")

    classifier = fit_algo(X_train, X_test, y_train, y_test, "Ridge")

    k = KFold(n_splits=5, shuffle=True, random_state=42)
    print(cross_val_score(classifier, X, y, cv=k),
          classifier.predict(X_today), sep="\t")

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    l1 = plt.plot(y_test, 'b--', linewidth=1, label="Real")
    l2 = plt.plot(classifier.predict(X_test), 'o--',
                  linewidth=1, )
    plt.setp(l1, label='Real', color='b', lw=1, ls='-', marker='+', ms=1.5)
    plt.setp(l2, label='Prediction', color='r',
             lw=1, ls='--', marker='o', ms=1.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
