'''
笔记
----------
一些常用的函数
'''
import numpy as np
import os
import re


def check_ddec_convergence(filename):
    '''
    功能
    ----------
    检查ddec的VASP_DDEC_analysis.output文件是否收敛

    参数
    ----------
    filename: VASP_DDEC_analysis.output文件名

    返回值
    ----------
    bool: True表示已经收敛，False表示未收敛。
    '''
    if not os.path.exists(filename):
        return False
    f_output = open(filename, 'r')
    while True:
        line = f_output.readline()
        if not line:
            break
        search = re.search(r"Finished chargemol", line)
        if search:
            return True
    return False


def check_vasp_convergence(filename):
    '''
    功能
    ----------
    检查vasp的OUTCAR文件是否收敛

    参数
    ----------
    filename: OUTCAR文件名

    返回值
    ----------
    bool: True表示已经收敛，False表示未收敛。
    '''
    if not os.path.exists(filename):
        return False
    f_outcar = open(filename, 'r')
    while True:
        line = f_outcar.readline()
        if not line:
            break
        search = re.search(r"General timing and accounting informations for this job", line)
        if search:
            return True
    return False


def diff_atoms(atoms1, atoms2, sorted=False, tolerance=1e-6):
    '''
    功能
    ----------
    检查两个ASE的atoms对象(单帧)是否相同

    参数
    ----------
    atoms1: ASE的atoms对象(单帧)
    atoms2: ASE的atoms对象(单帧)
    sorted: 比较之前是否需要对原子顺序按x坐标大小进行排序
    tolerance: 判断ASE的atoms对象(单帧)是否相同的tolerance

    返回值
    ----------
    bool: True表示相同，False表示不同
    '''
    distances1 = atoms1.get_all_distances(mic=True, vector=False)
    distances2 = atoms2.get_all_distances(mic=True, vector=False)
    distances_diff = np.sum(np.abs(distances1-distances2))
    if distances_diff < tolerance:
        return True
    else:
        return False


def mkdir(path):
    '''
    功能
    ----------
    模仿Shell里的mkdir

    参数
    ----------
    path: 文件夹路径

    返回值
    ----------
    无
    '''
    os.makedirs(path, exist_ok=True)


def seq(*args):
    '''
    功能
    ----------
    模仿Shell里的seq

    参数
    ----------
    一个两个或者三个数

    返回值
    ----------
    numpy array
    '''
    if len(args) == 1:
        return np.arange(1, args[0]+1)
    elif len(args) == 2:
        return np.arange(args[0], args[1]+1)
    elif len(args) == 3:
        return np.arange(args[0], args[2]+args[1], args[1])
    else:
        print('error using mytool.myfun.seq!!!')
        exit('1')
