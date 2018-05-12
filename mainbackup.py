# samples = int(input("Number of Samples: "))
# X, y = make_regression(n_samples=samples)
# bunch = load_boston(return_X_y=True)
# X = np.array(bunch[0], dtype=np.float64)
# y = np.array(bunch[1], dtype=np.float64)

# a, b, BigArray = stk.csv_to_df()
# BigArray = np.array(BigArray)
# print(BigArray)
# BigArray = BigArray[1:, :]
# collength = BigArray.shape[0]
# y = BigArray[:, 0].reshape(-1, 1)
# X = BigArray[:, 1].reshape(-1, 1)
# X = np.c_[X, np.arange(0, collength).reshape(-1, 1)]
#2337, 2330, 6223, 6220
    X_close, X_index, X_close_index, y_close, y_index, y_close_index = ProperReturn(
        6223)
    X = X_close_index
    y = y_close_index
    #print(X, y, sep="\n")

    X = nonlin_comp(X)
    X = poly_pre(X, 5)

    # print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, stratify=None, test_size=0.25, random_state=42)

    X_today = X[-1:, :].reshape(-1, X.shape[1])

    X_train, X_test, X_today = scale_data(
        X_train, X_test, X_today, "StandardScaler")

    # classifier = fit_neural_network(
    #    X_train, X_test, y_train.ravel(), y_test.ravel(), network_structure=(10, 10), activation="relu")
    classifier = fit_algo(X_train, X_test, y_train,
                          y_test, algotype="RidgeCV")

    #k = KFold(n_splits=5, shuffle=False, random_state=42)
    # print(np.mean(cross_val_score(classifier, X_test, y_test, cv=k)),
    #      classifier.predict(X_today), sep="\t")

    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    plot(classifier, X_train, y_train, "train plot")
    plot(classifier, X_test, y_test, "test plot")
