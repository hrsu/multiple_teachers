import matplotlib.pyplot as plt
from numpy import linspace
from scipy.stats import gaussian_kde, entropy

a = [8.44691, 8.234299, 7.1106877, 6.6839433, 6.0701976, 6.00876, 5.5366077,
     5.4475336, 5.161734, 5.040331, 4.8808165, 4.76408, 4.6493454, 4.5378766,
     4.43311, 4.3264365, 4.226094, 4.125937, 4.028902, 3.9315557, 3.8376522]

a_3 =  [8.452764, 8.306354, 8.105503, 7.7768083, 7.5999603, 6.98133, 7.070333,
        6.7929106, 6.523851, 6.5505676, 6.192373, 6.1669354, 6.092524, 5.792165,
        5.8032722, 5.7415857, 5.7375336, 5.401658, 5.2877235, 5.2108917, 5.11248]

a_1 = [8.444464, 8.118826, 7.7273245, 6.824777, 6.6612015, 6.8202143, 6.3021035,
       5.9515843, 5.7400446, 5.586173, 5.7613907, 5.349381, 5.347672, 5.161031,
       5.049584, 4.823991, 4.6123853, 4.5423245, 4.493977, 4.6546564, 4.2905807]


#svr处理
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

    plt.scatter(x, y, color='red')
    plt.plot(x, regressor.predict(x), color='blue')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.show()

    return y_pred

#kl散度
def cal_kl(m, n):
    n1 = len(m)
    n2 = len(n)
    m1 = min(min(m), min(n))
    m2 = max(max(m), max(n))
    lin = linspace(m1, m2, max(n1, n2))
    # 使用高斯kde估计pdf，Pdf:概率密度函数
    pdf1 = gaussian_kde(m)
    pdf2 = gaussian_kde(n)
    p = pdf1.pdf(lin)
    q = pdf2.pdf(lin)
    return entropy(p, q)

kl = cal_kl(list(SVR(a)), list(SVR(a_3)))
print("加噪30%后与原数据的KL: ", kl)

k2 = cal_kl(list(SVR(a)), list(SVR(a_1)))
print("加噪10%后与原数据的KL: ",k2)
