#K-S检验
#检验一个数列书否服从正态分布
# 两个数列是否服从相同分布
#链接：https://www.cnblogs.com/chaosimple/p/4090456.html


from scipy.stats import kstest
from scipy.stats import ks_2samp
import numpy as np

def tst_norm(x):
    '''
    检验是否满足正态分布
    '''
    test_stat = kstest(x, 'norm')
    print(test_stat)
    if test_stat[1]<=0.2:
        return False
    else:
        return True

def MaxMinNormalization(x):
    Min = min(x)
    Max = max(x)
    for i in range(len(x)):
        x[i] = (x[i] - Min) / (Max - Min)
    return x

def tst_samp(a,b):
    '''
    检验指定两个数列是否满足相同分布
    '''

    mean_x = np.mean(a)
    mean_y = np.mean(b)

    x = MaxMinNormalization(a)
    y = MaxMinNormalization(b)

    test_samp = ks_2samp(x,y)

    abs_rate = abs(mean_x-mean_y)/abs(mean_x)
    print("abs_rate: ", abs_rate)

    if abs_rate <=0.2:
        if test_samp[1] <= 0.2:
            print('They got different normal distribution!')
            print(test_samp[1])
            return False
        else:
            return True
    else:
        print('They got high change!')
        print(abs_rate)
        return False



