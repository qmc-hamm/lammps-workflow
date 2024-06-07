'''
笔记
----------
一些常用的数学工具
'''
import numpy as np


def cal_derivative(array_a, array_b):
    '''
    功能
    ----------
    计算导数

    参数
    ----------
    array_a: numpy array
    array_b: numpy array

    返回值
    ----------
    numpy array
    '''
    derivative = np.zeros(0)
    for i in range(1, len(array_a)-1):
        derivative = np.append(derivative, (array_b[i+1]-array_b[i-1])/(array_a[i+1]-array_a[i-1]))
    return derivative


def cal_mae(array_a, array_b):
    '''
    功能
    ----------
    计算Mean Absolute Error

    参数
    ----------
    array_a: numpy array
    array_b: numpy array

    返回值
    ----------
    float
    '''
    return np.mean(np.abs(array_a - array_b))


def cal_rmse(array_a, array_b):
    '''
    功能
    ----------
    计算Root Mean Squared Error

    参数
    ----------
    array_a: numpy array
    array_b: numpy array

    返回值
    ----------
    float
    '''
    return np.sqrt(np.mean(np.square(np.array(array_a)-np.array(array_b))))


def cal_r2(array_a, array_b):
    '''
    功能
    ----------
    计算R2

    参数
    ----------
    array_a: numpy array
    array_b: numpy array

    返回值
    ----------
    float
    '''
    return np.corrcoef([array_a, array_b])[0][1]


def cal_se(array_a):
    '''
    功能
    ----------
    计算Standard Error

    参数
    ----------
    array_a: numpy array
    array_b: numpy array

    返回值
    ----------
    float
    '''
    return np.std(array_a)/np.sqrt(len(array_a)-1)
