"""
笔记
----------
一些常用的建模函数
"""
import numpy as np
import os
import re
from mytool import myconstant


def build_orth_silicene(bond_length=2.27184,zigzag=1,armchair=2):
    """
    功能
    ----------
    生成Silicene结构

    参数
    ----------
    bond_length:Si-Si键长
    zigzag:zigzag方向六元环个数
    armchair:armchair方向六元环个数(必须为偶数)

    返回值
    ----------
    ASE中的atoms对象(单帧)
    """
    from ase import Atoms
    from ase import io
    atoms=io.read("%s/models/silicene.cif"%myconstant.path2mysharelib,index="0",format="cif")
    atoms=atoms.repeat((2,2,1))
    cell=atoms.get_cell()
    positions=atoms.get_positions()
    positions_select=np.zeros((0,3))
    for i in range(len(positions)):
        if positions[i][0]>=0.0 and positions[i][0]<cell[0][0]/2.0:
            if positions[i][1]>=0.0 and positions[i][1]<cell[0][0]*np.sqrt(3.0)/2.0:
                positions_select=np.r_[positions_select,[positions[i]]]
    positions_select*=bond_length/2.27184
    cell*=bond_length/2.27184
    atoms=Atoms(symbols="Si%d"%len(positions_select),positions=positions_select,cell=np.array([cell[0][0]/2.0,cell[0][0]*np.sqrt(3.0)/2.0,cell[2][2]]),pbc=True)
    atoms=atoms.repeat((zigzag,armchair//2,1))
    return atoms
