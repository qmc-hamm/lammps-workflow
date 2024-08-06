"""
笔记
----------
一些常用的io函数
"""
from io import StringIO
import numpy as np
import os
import pandas as pd
import re
import shutil
import subprocess as sp
from mytool import myconstant
from mytool import myfun
from ase import io
from ase import Atoms
from ase.calculators import calculator


def atoms2data(atoms,f_data,ele,style="atomic",bond_list=None,boxorth=False):
    """
    功能
    ----------
    将atoms(单帧)转为LAMMPS data文件

    参数
    ----------
    atoms: ASE中的atoms对象
    f_data: LAMMPS dada文件名
    ele: 元素对应的data文件type顺序
    style: LAMMPS data文件的style, 比如atomic, charge, full
    bond_list: bond list to write, optional
    boxorth: 是否需要是正交盒子

    返回值
    ----------
    无
    """
    def get_mol_id(atom_id, bond_list):
        if len(np.where(bond_list[:, 1:] == atom_id)[0]) == 0:
            return -1
        else:
            return np.where(bond_list[:, 1:] == atom_id)[0][0]
    f_output = open(f_data, "w")
    cell = np.array(atoms.get_cell())
    volume = atoms.get_volume()
    vec_A = cell[0]
    vec_B = cell[1]
    vec_C = cell[2]
    lx = np.linalg.norm(vec_A)
    xy = np.inner(vec_B, vec_A/np.linalg.norm(vec_A))
    ly = np.sqrt(np.linalg.norm(vec_B)**2-xy**2)
    xz = np.inner(vec_C, vec_A/np.linalg.norm(vec_A))
    yz = (np.inner(vec_B, vec_C)-xy*xz)/ly
    lz = np.sqrt(np.linalg.norm(vec_C)**2-xz**2-yz**2)
    positions = atoms.get_positions()
    positions = np.dot(positions, np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]]))
    positions /= volume
    positions = np.dot(positions, np.array([np.cross(vec_B, vec_C), np.cross(vec_C, vec_A), np.cross(vec_A, vec_B)]))
    chemical_symbols = atoms.get_chemical_symbols()
    chemical_symbols_list = list()
    atom_types = list()
    for j in range(len(chemical_symbols)):
        if chemical_symbols[j] not in chemical_symbols_list:
            chemical_symbols_list.append(chemical_symbols[j])
        atom_types.append(ele.index(chemical_symbols[j])+1)
    f_output.write("data.txt (written with https://github.com/hongweiniu/mysharelib)\n\n")
    f_output.write("%d atoms\n" % len(positions))
    if bond_list is not None:
        f_output.write("%d bonds\n" % len(bond_list))
    f_output.write("%d atom types\n" % len(ele))
    if bond_list is not None:
        f_output.write("%d bond types\n" % len(np.unique(bond_list[:, 0])))
    f_output.write("\n")
    f_output.write("0.0 %f xlo xhi\n" % lx)
    f_output.write("0.0 %f ylo yhi\n" % ly)
    f_output.write("0.0 %f zlo zhi\n" % lz)
    if boxorth == False:
        f_output.write("%f %f %f xy xz yz\n\n" % (xy, xz, yz))
    f_output.write("Atoms # %s\n\n" % style)
    if style == "atomic":
        for i in range(len(positions)):
            f_output.write("%d %d %f %f %f\n" % (i+1, atom_types[i], positions[i][0], positions[i][1], positions[i][2]))
    if style == "charge":
        charges = atoms.get_initial_charges()
        for i in range(len(positions)):
            f_output.write("%d %d %f %f %f %f\n" % (i+1, atom_types[i], charges[i], positions[i][0], positions[i][1], positions[i][2]))
    if style == "full":
        charges = atoms.get_initial_charges()
        for i in range(len(positions)):
            f_output.write("%d %d %d %f %f %f %f\n" % (i+1, get_mol_id(i, bond_list)+1, atom_types[i], charges[i], positions[i][0], positions[i][1], positions[i][2]))
        f_output.write("\nBonds\n\n")
        for i in range(len(bond_list)):
            f_output.write("%d %d %d %d\n" % (i+1, bond_list[i][0], bond_list[i][1]+1, bond_list[i][2]+1))
    f_output.close()


def atoms2dump(atoms, f_dump, ele, style="atomic"):
    """
    功能
    ----------
    将atoms(list of atoms)转为LAMMPS dump文件

    参数
    ----------
    atoms: ASE中的atoms对象
    f_dump: LAMMPS dump文件名
    ele: 元素对应的dump文件type顺序
    style: LAMMPS dump文件的style, 比如atomic, charge

    返回值
    ----------
    无
    """
    f_output = open(f_dump, "w")
    for i in range(len(atoms)):
        cell = np.array(atoms[i].get_cell())
        volume = atoms[i].get_volume()
        vec_A = cell[0]
        vec_B = cell[1]
        vec_C = cell[2]
        lx = np.linalg.norm(vec_A)
        xy = np.inner(vec_B, vec_A/np.linalg.norm(vec_A))
        ly = np.sqrt(np.linalg.norm(vec_B)**2-xy**2)
        xz = np.inner(vec_C, vec_A/np.linalg.norm(vec_A))
        yz = (np.inner(vec_B, vec_C)-xy*xz)/ly
        lz = np.sqrt(np.linalg.norm(vec_C)**2-xz**2-yz**2)
        positions = atoms[i].get_positions()
        positions = np.dot(positions, np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]]))
        positions /= volume
        positions = np.dot(positions, np.array([np.cross(vec_B, vec_C), np.cross(vec_C, vec_A), np.cross(vec_A, vec_B)]))
        chemical_symbols = atoms[i].get_chemical_symbols()
        chemical_symbols_list = list()
        atom_types = list()
        for j in range(len(chemical_symbols)):
            if chemical_symbols[j] not in chemical_symbols_list:
                chemical_symbols_list.append(chemical_symbols[j])
            atom_types.append(ele.index(chemical_symbols[j])+1)
        f_output.write("ITEM: TIMESTEP\n")
        f_output.write("%d\n" % i)
        f_output.write("ITEM: NUMBER OF ATOMS\n")
        f_output.write("%d\n" % len(positions))
        f_output.write("ITEM: BOX BOUNDS xy xz yz ")
        f_output.write("pp pp pp\n")
        xlo_bound = 0.0 + np.min([0.0, xy, xz, xy+xz])
        xhi_bound = lx + np.max([0.0, xy, xz, xy+xz])
        ylo_bound = 0.0 + np.min([0.0, yz])
        yhi_bound = ly + np.max([0.0, yz])
        zlo_bound = 0.0
        zhi_bound = lz
        f_output.write("%f %f %f\n" % (xlo_bound, xhi_bound, xy))
        f_output.write("%f %f %f\n" % (ylo_bound, yhi_bound, xz))
        f_output.write("%f %f %f\n" % (zlo_bound, zhi_bound, yz))
        if style == "atomic":
            f_output.write("ITEM: ATOMS id type x y z\n")
            for j in range(len(positions)):
                f_output.write("%d %d %f %f %f\n" %(j+1, atom_types[j], positions[j][0], positions[j][1], positions[j][2]))
        if style == "charge":
            f_output.write("ITEM: ATOMS id type q x y z\n")
            charges = atoms[i].get_initial_charges()
            for j in range(len(positions)):
                f_output.write("%d %d %f %f %f %f\n" %(j+1, atom_types[j], charges[j], positions[j][0], positions[j][1], positions[j][2]))
    f_output.close()


def atoms2ipixyz(atoms, f_ipixyz):
    """
    功能
    ----------
    将atoms(单帧)转为ipi xyz文件

    参数
    ----------
    atoms: ASE中的atoms对象
    f_ipixyz: ipi xyz文件

    返回值
    ----------
    无
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell_lengths_and_angles()
    symbols = atoms.get_chemical_symbols()
    system_size = len(positions)
    file_ipixyz = open(f_ipixyz, "w")
    file_ipixyz.write("%d\n" % system_size)
    file_ipixyz.write("# CELL(abcABC): ")
    for i in range(6):
        file_ipixyz.write("%f " % cell[i])
    file_ipixyz.write("Traj: positions{angstrom} Step: 0 Bead: 0 cell{angstrom}\n")
    for i in range(system_size):
        file_ipixyz.write("%s " % symbols[i])
        for j in range(3):
            file_ipixyz.write("%f " % positions[i][j])
        file_ipixyz.write("\n")
    file_ipixyz.close()


def atoms2raw(atoms,d_raw=".",virial=False):
    """
    功能
    ----------
    将atoms(list)转为DeePMD的raw文件

    参数
    ----------
    atoms: ASE中的atoms对象
    d_raw: 生成raw文件的路径
    virial: 是否生成virial信息

    返回值
    ----------
    无
    """
    system_size = len(atoms[0])
    box = np.zeros((len(atoms), 9))
    energy = np.zeros(len(atoms))
    positions = np.zeros((len(atoms), system_size*3))
    forces = np.zeros((len(atoms), system_size*3))
    for i in range(len(atoms)):
        for j in range(3):
            box[i][3*j:3*j+3] = atoms[i].get_cell()[j]
        energy[i] = atoms[i].get_potential_energy()
        positions[i] = atoms[i].get_positions().flatten(order="C")
        forces[i] = atoms[i].get_forces().flatten(order="C")
    os.makedirs(d_raw, exist_ok=True)
    np.savetxt("%s/box.raw" % d_raw, box)
    np.savetxt("%s/energy.raw" % d_raw, energy)
    np.savetxt("%s/coord.raw" % d_raw, positions)
    np.savetxt("%s/force.raw" % d_raw, forces)
    if virial == True:
        virials = np.zeros((len(atoms), 9))
        for i in range(len(atoms)):
            virials_temp=atoms[i].get_stress()/myconstant.ev_angstrom_3_2_gpa*atoms[i].get_volume()
            virials[i][0] = virials_temp[0] # XX
            virials[i][1] = virials_temp[5] # XY
            virials[i][2] = virials_temp[4] # XZ
            virials[i][3] = virials_temp[5] # YX
            virials[i][4] = virials_temp[1] # YY
            virials[i][5] = virials_temp[3] # YZ
            virials[i][6] = virials_temp[4] # ZX
            virials[i][7] = virials_temp[3] # ZY
            virials[i][8] = virials_temp[2] # ZZ
        np.savetxt("%s/virial.raw" % d_raw, virials)
    symbols = atoms[0].get_chemical_symbols()
    type_map = list()
    for i in symbols:
        if i not in type_map:
            type_map.append(i)
    f=open("%s/type_map.raw"%d_raw,"w")
    for i in type_map:
        f.write("%s\n"%i)
    f.close()
    type_array = np.zeros(system_size, dtype=int)
    for i in range(len(symbols)):
        for j in range(len(type_map)):
            if symbols[i] == type_map[j]:
                type_array[i] = j
    np.savetxt("%s/type.raw" % d_raw, type_array, fmt="%d")


def atoms2vasp(atoms, f_vasp, ele, direct=False):
    """
    功能
    ----------
    将atoms(单帧)转为vasp的结构文件(POSCAR, CONTCAR)。这个功能其实ase也有，但ase不会自动把所有原子按元素种类排序。

    参数
    ----------
    atoms: ASE中的atoms对象(单帧)
    f_vasp: 生成vasp的结构文件(POSCAR, CONTCAR)路径
    ele: 元素列表
    direct: 是否为分数坐标

    返回值
    ----------
    无
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    symbols_sort = list()
    positions_sort = np.zeros([len(positions), 3])
    counter = 0
    for i in range(len(ele)):
        for j in range(len(symbols)):
            if symbols[j] == ele[i]:
                symbols_sort.append(symbols[j])
                positions_sort[counter] = positions[j]
                counter += 1
    atoms_sort = Atoms(symbols=symbols_sort, positions=positions_sort, cell=cell, pbc=True)
    io.write(f_vasp, atoms_sort, format="vasp", direct=direct)


def data2atoms(f_data, ele, style):
    """
    功能
    ----------
    将LAMMPS data文件转为atoms(单帧)

    参数
    ----------
    f_data: LAMMPS dada文件名
    ele: 元素列表
    style: LAMMPS data文件的style, 比如atomic, charge

    返回值
    ----------
    ASE中的atoms对象(单帧)
    """
    f_input = open(f_data, "r")
    system_size = 0
    xlo = 0.0
    xhi = 0.0
    ylo = 0.0
    yhi = 0.0
    zlo = 0.0
    zhi = 0.0
    if style == "full":
        while True:
            line = f_input.readline()
            if not line:
                break
            search = re.search(r"(\d+)\s+atoms", line)
            if search:
                system_size = int(search.group(1))
            search = re.search(r"(\S+)\s+(\S+)\s+xlo\s+xhi", line)
            if search:
                xlo = float(search.group(1))
                xhi = float(search.group(2))
            search = re.search(r"(\S+)\s+(\S+)\s+ylo\s+yhi", line)
            if search:
                ylo = float(search.group(1))
                yhi = float(search.group(2))
            search = re.search(r"(\S+)\s+(\S+)\s+zlo\s+zhi", line)
            if search:
                zlo = float(search.group(1))
                zhi = float(search.group(2))
        f_input.seek(0)
        types = np.zeros(system_size, dtype=int)
        positions = np.zeros((system_size, 3))
        while True:
            line = f_input.readline()
            if not line:
                break
            search = re.search(r"Atoms\n", line)
            if search:
                line = f_input.readline()
                for i in range(system_size):
                    line = f_input.readline().split()
                    for j in range(3):
                        types[i] = int(line[2])
                        positions[i][j] = float(line[4+j])
        symbols = [None]*system_size
        for i in range(len(types)):
            symbols[i] = ele[types[i]-1]
        atoms = Atoms(symbols=symbols, positions=positions, cell=np.array([xhi-xlo, yhi-ylo, zhi-zlo]), pbc=True)
        f_input.close()
        return atoms
    if style == "atomic":
        while True:
            line = f_input.readline()
            if not line:
                break
            search = re.search(r"(\d+)\s+atoms", line)
            if search:
                system_size = int(search.group(1))
            search = re.search(r"(\S+)\s+(\S+)\s+xlo\s+xhi", line)
            if search:
                xlo = float(search.group(1))
                xhi = float(search.group(2))
            search = re.search(r"(\S+)\s+(\S+)\s+ylo\s+yhi", line)
            if search:
                ylo = float(search.group(1))
                yhi = float(search.group(2))
            search = re.search(r"(\S+)\s+(\S+)\s+zlo\s+zhi", line)
            if search:
                zlo = float(search.group(1))
                zhi = float(search.group(2))
        f_input.seek(0)
        types = np.zeros(system_size, dtype=int)
        positions = np.zeros((system_size, 3))
        while True:
            line = f_input.readline()
            if not line:
                break
            search = re.search(r"Atoms", line)
            if search:
                while True:
                    line = f_input.readline().split()
                    if len(line)!=6:
                        continue
                    types[0] = int(line[1])
                    for j in range(3):
                        positions[0][j] = float(line[3+j])
                    for i in range(1,system_size):
                        line = f_input.readline().split()
                        types[i] = int(line[1])
                        for j in range(3):
                            positions[i][j] = float(line[3+j])
                    break
        symbols = [None]*system_size
        for i in range(len(types)):
            symbols[i] = ele[types[i]-1]
        atoms = Atoms(symbols=symbols, positions=positions, cell=np.array([xhi-xlo, yhi-ylo, zhi-zlo]), pbc=True)
        f_input.close()
        return atoms
    if style == "charge":
        while True:
            line = f_input.readline()
            if not line:
                break
            search = re.search(r"(\d+)\s+atoms", line)
            if search:
                system_size = int(search.group(1))
            search = re.search(r"(\S+)\s+(\S+)\s+xlo\s+xhi", line)
            if search:
                xlo = float(search.group(1))
                xhi = float(search.group(2))
            search = re.search(r"(\S+)\s+(\S+)\s+ylo\s+yhi", line)
            if search:
                ylo = float(search.group(1))
                yhi = float(search.group(2))
            search = re.search(r"(\S+)\s+(\S+)\s+zlo\s+zhi", line)
            if search:
                zlo = float(search.group(1))
                zhi = float(search.group(2))
        f_input.seek(0)
        types = np.zeros(system_size, dtype=int)
        charges=np.zeros(system_size)
        positions = np.zeros((system_size, 3))
        while True:
            line = f_input.readline()
            if not line:
                break
            search = re.search(r"Atoms", line)
            if search:
                while True:
                    line = f_input.readline().split()
                    if len(line)!=6:
                        continue
                    types[0] = int(line[1])
                    charges[0]=float(line[2])
                    for j in range(3):
                        positions[0][j] = float(line[3+j])
                    for i in range(1,system_size):
                        line = f_input.readline().split()
                        types[i] = int(line[1])
                        charges[i]=float(line[2])
                        for j in range(3):
                            positions[i][j] = float(line[3+j])
                    break
        symbols = [None]*system_size
        for i in range(len(types)):
            symbols[i] = ele[types[i]-1]
        atoms = Atoms(symbols=symbols, positions=positions, cell=np.array([xhi-xlo, yhi-ylo, zhi-zlo]), pbc=True, charges=charges)
        f_input.close()
        return atoms


def dump2atoms(f_dump, ele, index=":"):
    """
    功能
    ----------
    将LAMMPS dump文件转为atoms对象(list)

    参数
    ----------
    f_dump: LAMMPS dump文件名
    ele: 元素对应的dump文件type顺序
    index: ase.io中的index

    返回值
    ----------
    ASE中的atoms对象(list)
    """
    atoms = io.read(f_dump, index=index, format="lammps-dump-text")
    for i in range(len(atoms)):
        chemical_symbols = atoms[i].get_chemical_symbols()
        chemical_symbols_mod = list()
        for j in chemical_symbols:
            chemical_symbols_mod.append(ele[myconstant.periodic_table.index(j)])
        atoms[i].set_chemical_symbols(chemical_symbols_mod)
    return atoms


def outcar2atoms(f_outcar):
    """
    功能
    ----------
    将outcar文件转为atoms(list). 读取能量(eV), 力(ev/Angstrom), 应力(GPa, order: xx, yy, zz, yz, xz, xy)

    参数
    ----------
    f_outcar: outcar文件名
    返回值
    ----------
    ASE中的atoms对象(list)
    """
    from ase.calculators.singlepoint import SinglePointCalculator as SPC
    type_num=list()
    with open(f_outcar,"r") as filename:
        while True:
            line=filename.readline()
            if not line:
                break
            search=re.search(r"ions per type =",line)
            if search:
                pattern=re.compile(r"\d+")
                result=pattern.findall(line)
                for i in result:
                    type_num.append(int(i))
                break
    # print(type_num)
    ele=list()
    with open(f_outcar,"r") as filename:
        while True:
            line=filename.readline()
            if not line:
                break
            search=re.search(r"INCAR:",line)
            if search:
                for i in range(len(type_num)):
                    line=filename.readline()
                    ele.append(re.search(r"POTCAR:\s+\S+\s+(\S+)\s+\S+",line).group(1).split("_")[0])
                break
    # print(ele)
    system_size=np.sum(type_num)
    atoms=list()
    with open(f_outcar,"r") as filename:
        while True:
            line=filename.readline()
            if not line:
                break
            search=re.search(r"aborting loop because EDIFF is reached",line)
            early_termination=0
            if search:
                stress=np.zeros(6)
                cell=np.zeros((3,3))
                positions=np.zeros((system_size,3))
                forces=np.zeros((system_size,3))
                energy=0.0
                while True:
                    line=filename.readline()
                    if not line:
                        early_termination=1
                        break
                    search=re.search(r"in\s+kB",line)
                    if search:
                        # vasp stress is in kB, vasp order: XX, YY, ZZ, XY, YZ, ZX, ase order: xx, yy, zz, yz, xz, xy
                        stress_read=myconstant.pattern_real.findall(line)
                        stress[0]=float(stress_read[0][0])
                        stress[1]=float(stress_read[1][0])
                        stress[2]=float(stress_read[2][0])
                        stress[3]=float(stress_read[4][0])
                        stress[4]=float(stress_read[5][0])
                        stress[5]=float(stress_read[3][0])
                        stress*=0.1 # to GPa
                        # print(stress)
                        break
                while True:
                    line=filename.readline()
                    if not line:
                        early_termination=1
                        break
                    search=re.search(r"direct lattice vectors",line)
                    if search:
                        for i in range(3):
                            line=filename.readline()
                            search=re.search(r"(\S+)\s+(\S+)\s+(\S+)",line)
                            for j in range(3):
                                cell[i][j]=float(search.group(j+1))
                        # print(cell)
                        break
                while True:
                    line=filename.readline()
                    if not line:
                        early_termination=1
                        break
                    search=re.search(r"POSITION\s+TOTAL-FORCE",line)
                    if search:
                        line=filename.readline()
                        for i in range(system_size):
                            line=filename.readline()
                            search=re.search(r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",line)
                            for j in range(3):
                                positions[i][j]=float(search.group(j+1))
                                forces[i][j]=float(search.group(j+4))
                        # print(positions)
                        # print(forces)
                        break
                while True:
                    line=filename.readline()
                    if not line:
                        early_termination=1
                        break
                    search=re.search(r"free\s+energy\s+TOTEN\s+=\s+(\S+)\s+eV",line)
                    if search:
                        energy=float(search.group(1))
                        # print(energy)
                        break
                if early_termination==1:
                    break
                symbols=""
                for i in range(len(type_num)):
                    symbols+=ele[i]
                    symbols+="%d"%type_num[i]
                # print(symbols)
                atoms_append=Atoms(symbols=symbols,positions=positions,cell=cell,pbc=True)
                calc=SPC(atoms=atoms_append,energy=energy,forces=forces,stress=stress)
                atoms_append.set_calculator(calc)
                atoms.append(atoms_append)
    return atoms


def ipixyz2atoms(f_ipixyz,f_velo=None):
    """
    功能
    ----------
    将ipi xyz文件转为atoms(list)

    参数
    ----------
    f_ipixyz: ipi xyz文件(长度单位为埃米)
    f_velo: ipi 速度文件(速度单位为原子单位),可选

    返回值
    ----------
    ASE中的atoms对象(list)
    """
    cell = np.zeros([0, 6])
    f = open(f_ipixyz, "r")
    while True:
        line = f.readline()
        if not line:
            break
        search = re.search(r"CELL\(abcABC\):\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if search:
            cell_append = np.zeros([1, 6])
            for i in range(6):
                cell_append[0][i] = float(search.group(i+1))
            cell = np.r_[cell, cell_append]
    f.close()
    atoms = io.read(f_ipixyz, format="extxyz", index=":")
    positions = [atoms[i].get_positions() for i in range(len(atoms))]
    if f_velo!=None:
        atoms_velo=io.read(f_velo,format="extxyz",index=":")
        velocities=[atoms_velo[i].get_positions() for i in range(len(atoms_velo))]
        for i in range(len(atoms)):
            atoms[i].set_positions(positions[i])
            atoms[i].set_velocities(velocities[i])
            # print(cell[i])
            atoms[i].set_cell(cell[i])
            atoms[i].set_pbc((1,1,1))
            atoms[i].wrap()
    else:
        for i in range(len(atoms)):
            atoms[i].set_positions(positions[i])
            # print(cell[i])
            atoms[i].set_cell(cell[i])
            atoms[i].set_pbc((1,1,1))
            atoms[i].wrap()
    return atoms


def read_bond_list_from_data(f_data):
    """
    功能
    ----------
    从LAMMPS data文件读bond_list(如果有)

    参数
    ----------
    f_data: LAMMPS dada文件名

    返回值
    ----------
    bond_list(如果有)
    """
    bond_list = np.zeros((0, 3), dtype=int)
    f_input = open(f_data, "r")
    while True:
        line = f_input.readline()
        if not line:
            break
        search = re.search(r"Bonds", line)
        if search:
            while True:
                line = f_input.readline()
                if not line:
                    return bond_list
                search = re.search(r"\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+", line)
                if search:
                    bond_list_append = np.zeros((1, 3), dtype=int)
                    for i in range(3):
                        bond_list_append[0][i] = search.group(i+1)
                    bond_list = np.r_[bond_list, bond_list_append]
    f_input.close()


def read_lammps_thermo(filename):
    """
    功能
    ----------
    读LAMMPS的thermo文件

    参数
    ----------
    filename: 文件名

    返回值
    ----------
    Pandas的DataFrame
    """
    f_lammps_out = open(filename, "r")
    str_lammps_out = ""
    counter = 0
    while True:
        line = f_lammps_out.readline()
        if not line:
            break
        search = re.search("Step", line)
        if search:
            if counter == 0:
                str_lammps_out += line
                counter += 1
            while True:
                line = f_lammps_out.readline()
                if not line:
                    break
                search1 = re.search(r"[(\-|\+)?\d+(\.\d+)?\s+]+\n$", line)
                search2 = re.search(r"Loop time of", line)
                search3 = re.search(r"[A-Za-df-z]", line)
                if search1 and not search3:
                    str_lammps_out += line
                if search2:
                    break
    if str_lammps_out == "":
        return pd.DataFrame()
    data = StringIO(str_lammps_out)
    data_frame = pd.read_csv(data, sep=r"\s+")
    return data_frame


def read_vasp_energy(filename):
    """
    功能
    ----------
    读vasp的OUTCAR文件中的能量

    参数
    ----------
    filename: 文件名

    返回值
    ----------
    float
    """
    f_outcar = open(filename, "r")
    energy = 0.0
    while True:
        line = f_outcar.readline()
        if not line:
            break
        search = re.search(r"energy\(sigma->0\)\s+=\s+(\S+)", line)
        if search:
            energy = float(search.group(1))
    return energy


def read_vasp_temperature(filename):
    """
    功能
    ----------
    读vasp的OUTCAR文件中的温度(针对aimd计算)

    参数
    ----------
    filename: 文件名

    返回值
    ----------
    numpy array
    """
    f_outcar = open(filename, "r")
    temperature = list()
    while True:
        line = f_outcar.readline()
        if not line:
            break
        search = re.search(r"kin.\s+lattice\s+EKIN_LAT=\s+\S+\s+\(temperature\s+(\S+)", line)
        if search:
            temperature.append(float(search.group(1)))
    return temperature


def set2atoms(dir_in):
    """
    功能
    ----------
    将DeePMD的数据集转为atoms(list)

    参数
    ----------
    dir_in:将DeePMD的数据集文件夹

    返回值
    ----------
    ASE中的atoms对象(list)
    """
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator as SPC
    atoms=list()
    box_all=np.load("%s/set.000/box.npy"%dir_in)
    coord_all=np.load("%s/set.000/coord.npy"%dir_in)
    energy_all=np.load("%s/set.000/energy.npy"%dir_in)
    force_all=np.load("%s/set.000/force.npy"%dir_in)
    type_file=np.loadtxt("%s/type.raw"%dir_in,dtype=int)
    type_map_file=np.loadtxt("%s/type_map.raw"%dir_in,dtype=str)
    symbols=list()
    # print(type_file)
    # print(type_map_file)
    for i in type_file:
        symbols.append(type_map_file[i])
    # print(symbols)
    for i in range(len(energy_all)):
        cell=box_all[i].reshape(-1,3)
        positions=coord_all[i].reshape(-1,3)
        energy=energy_all[i]
        forces=force_all[i].reshape(-1,3)
        atoms_append=Atoms(symbols=symbols,positions=positions,cell=cell,pbc=True)
        calc=SPC(atoms=atoms_append,energy=energy,forces=forces)
        atoms_append.set_calculator(calc)
        atoms.append(atoms_append)
    return atoms


def vasp_fix_atoms(file_in, file_out, fix_atoms):
    """
    功能
    ----------
    给POSCAR/CONTCAR文件部分原子加限制

    参数
    ----------
    file_in: 输入文件
    file_out: 输出文件
    fix_atoms: 原子被限制的情况

    返回值
    ----------
    无
    """
    f_in=open(file_in, "r")
    f_out=open(file_out, "w")
    for i in range(2):
        f_out.write(f_in.readline())
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = f_in.readline().split()
    for i in range(3):
        f_out.write("%s %s %s\n" % tuple(cell[i]))
    for i in range(2):
        f_out.write(f_in.readline())
    line=f_in.readline()
    if re.search("Selective dynamics",line):
        f_out.write(line)
    else:
        f_out.write("Selective dynamics\n")
        f_out.write(line)
    for i in range(len(fix_atoms)):
        f_out.write("%s %s %s %s %s %s\n"%(tuple(f_in.readline().split()[:3]+fix_atoms[i])))
    f_in.close()
    f_out.close()


def vasp_apply_strain(file_in, file_out, epsilon):
    """
    功能
    ----------
    给Direct坐标的POSCAR/CONTCAR文件加应变(改变盒子)

    参数
    ----------
    file_in: 输入文件
    file_out: 输出文件
    epsilon: 应变, 长度为6

    返回值
    ----------
    无
    """
    f_in = open(file_in, "r")
    f_out = open(file_out, "w")
    for i in range(2):
        f_out.write(f_in.readline())
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = f_in.readline().split()
    apply_strain = np.identity(3) + np.array([[epsilon[0], 0.5*epsilon[5], 0.5*epsilon[4]], [0.5*epsilon[5], epsilon[1], 0.5*epsilon[3]], [0.5*epsilon[4], 0.5*epsilon[3], epsilon[2]]])
    cell = np.dot(cell, apply_strain)
    for i in range(3):
        f_out.write("%s %s %s\n" % tuple(cell[i]))
    while True:
        line = f_in.readline()
        if not line:
            break
        else:
            f_out.write(line)
    f_in.close()
    f_out.close()


def write_deepmd_pot(dir_in,file_in,file_out):
    """
    功能
    ----------
    生成DeePMD势函数

    参数
    ----------
    dir_in: 输入文件所在路径
    file_in: 输入文件名字
    file_out: 输出文件

    返回值
    ----------
    无
    """
    pid=os.getpid()
    myfun.mkdir("%d"%pid)
    shutil.copy("%s/%s.meta"%(dir_in,file_in),"%d"%pid)
    shutil.copy("%s/%s.index"%(dir_in,file_in),"%d"%pid)
    shutil.copy("%s/%s.data-00000-of-00001"%(dir_in,file_in),"%d"%pid)
    file_checkpoint=open("%d/checkpoint"%pid,"w")
    file_checkpoint.write("model_checkpoint_path: \"%s/%d/%s\"\n"%(os.getcwd(),pid,file_in))
    file_checkpoint.write("all_model_checkpoint_paths: \"%s/%d/%s\""%(os.getcwd(),pid,file_in))
    file_checkpoint.close()
    os.chdir("%d"%pid)
    subp=sp.Popen("dp freeze -o ../%s"%file_out,shell=True)
    subp.wait()
    os.chdir("..")
    shutil.rmtree("%d"%pid)
