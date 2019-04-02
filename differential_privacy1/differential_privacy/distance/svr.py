def SVR(x):
    from sklearn.svm import SVR
    import numpy as np

    x = np.array(x)
    y = np.array(list(range(len(a)))).reshape(-1, 1)
    x, y = y, x

    # 回归
    regressor = SVR(kernel='rbf', C=1e3, gamma=0.01)
    regressor.fit(x, y)
    # y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
    # y_pred = sc_y.inverse_transform(y_pred)

    y_pred = regressor.predict(x)
    return y_pred
