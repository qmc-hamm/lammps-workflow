"""
笔记
----------
一些常用的计算
"""
import numpy as np
import os
import shutil
import subprocess as sp
from mytool import myconstant
from mytool import myfun
from mytool import myio
from mytool import mymath
from ase import Atoms
from ase import io
from ase.calculators import calculator


def cal_ef_error(atoms0,atoms1,eatom0=0.0,eatom1=0.0):
    """
    功能
    ----------
    计算两个含有能量和力信息的ase的Atoms对象之间的能量和力的误差(atoms1和atoms2帧数相同且其中结构原子数相同)

    参数
    ----------
    atoms0:第1个ase的Atoms对象
    atoms1:第2个ase的Atoms对象
    eatoms1:第1个ase的Atoms对象的每个原子的能量修正(单位:eV/atom)
    eatoms1:第2个ase的Atoms对象的每个原子的能量修正(单位:eV/atom)

    返回值
    ----------
    e_rmse,f_rmse(单位分别是meV/atom,meV/Angstrom)
    """
    system_size=len(atoms0[0].get_positions())
    e0=np.array([i.get_potential_energy() for i in atoms0])/system_size-eatom0
    e1=np.array([i.get_potential_energy() for i in atoms1])/system_size-eatom1
    f0=np.array([i.get_forces().ravel() for i in atoms0])
    f1=np.array([i.get_forces().ravel() for i in atoms1])
    e_rmse=1000*mymath.cal_rmse(e0,e1)
    f_rmse=1000*mymath.cal_rmse(f0,f1)
    return e_rmse,f_rmse


def cal_polycrystalline(ela_con):
    """
    功能
    ----------
    根据弹性常数矩阵计算致密的各向同性的系统多晶量

    参数
    ----------
    ela_con：numpy array, 6*6, 弹性常数矩阵

    返回值
    ----------
    k_vrh, g_vrh, mu, e
    """
    fle_mat = np.linalg.inv(ela_con)
    k_v = 1.0/9.0*((ela_con[0][0]+ela_con[1][1]+ela_con[2][2])+2*(ela_con[0][1]+ela_con[1][2]+ela_con[2][0]))
    k_r = 1.0/((fle_mat[0][0]+fle_mat[1][1]+fle_mat[2][2])+2*(fle_mat[0][1]+fle_mat[1][2]+fle_mat[2][0]))
    k_vrh = 0.5*(k_v+k_r)
    g_v = 1.0/15.0*((ela_con[0][0]+ela_con[1][1]+ela_con[2][2])-(ela_con[0][1]+ela_con[1][2]+ela_con[2][0])+3*(ela_con[3][3]+ela_con[4][4]+ela_con[5][5]))
    g_r = 15.0/(4*(fle_mat[0][0]+fle_mat[1][1]+fle_mat[2][2])-4*(fle_mat[0][1]+fle_mat[1][2]+fle_mat[2][0])+3*(fle_mat[3][3]+fle_mat[4][4]+fle_mat[5][5]))
    g_vrh = 0.5*(g_v+g_r)
    mu = (3*k_vrh-2*g_vrh)/(6*k_vrh+2*g_vrh)
    e = 9*k_vrh*g_vrh/(3*k_vrh+g_vrh)
    return k_vrh, g_vrh, mu, e


def cal_rdf(filename, end, num_of_confs, every, cutoff, num_of_bins, keyword="all"):
    """
    功能
    ----------
    计算rdf

    参数
    ----------
    filename: 输入结构文件名
    end: 最后一帧
    num_of_confs: 合计结构数
    every: 每隔#帧取一次结构
    cutoff: 截断半径
    num_of_bins: bin个数
    keyword: 关键词, 比如"all", "1-1", "1-2", "2-2"

    返回值
    ----------
    共3列。第一列是半径r，第二列是rdf，第三列是cutoff内合计原子数。
    """
    from ovito.modifiers import CoordinationAnalysisModifier
    from ovito.io import import_file
    delta = cutoff/num_of_bins
    pipeline = import_file(filename)
    r = np.linspace(delta/2, cutoff-delta/2, num=num_of_bins)
    total_rdf = np.zeros(num_of_bins)
    coord_num = np.zeros(num_of_bins)
    if keyword == "all":
        modifier = CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=num_of_bins)
        pipeline.modifiers.append(modifier)
        if end < 0:
            end += pipeline.source.num_frames+1
        for frame in range(end-(num_of_confs-1)*every, end+1, every):
            data = pipeline.compute(frame)
            current_rdf = data.tables["coordination-rdf"].xy()
            gr = current_rdf[:, 1]
            total_rdf += gr
            rho = data.particles.count / data.cell.volume
            coord_num_core = gr*4*np.pi*rho*r**2
            coord_num += np.array([np.trapz(coord_num_core[:k+1], dx=delta) for k in range(len(coord_num_core))])
    else:
        modifier = CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=num_of_bins, partial=True)
        pipeline.modifiers.append(modifier)
        if end < 0:
            end += pipeline.source.num_frames+1
        for frame in range(end-(num_of_confs-1)*every, end+1, every):
            data = pipeline.compute(frame)
            current_rdf = data.series["coordination-rdf"].xy()
            index = data.tables["coordination-rdf"].y.component_names.index(keyword)
            gr = current_rdf[:, index+1]
            total_rdf += gr
            rho = data.particles.count / data.cell.volume
            coord_num_core = gr*4*np.pi*rho*r**2
            coord_num += np.array([np.trapz(coord_num_core[:k+1], dx=delta) for k in range(len(coord_num_core))])
    total_rdf /= num_of_confs
    coord_num /= num_of_confs
    return np.column_stack((r, total_rdf, coord_num))


def cal_sf(filename, end, num_of_confs, every, cutoff, num_of_bins, k_cutoff, k_number_of_bins, keyword="all"):
    """
    功能
    ----------
    计算sf

    参数
    ----------
    filename: 输入结构文件名
    end: 最后一帧
    num_of_confs: 合计结构数
    every: 每隔#帧取一次结构
    cutoff: 截断半径
    num_of_bins: bin个数
    k_cutoff: k的截断半径
    k_number_of_bins: k的bin个数
    keyword: 关键词, 比如"all", "1-1", "1-2", "2-2"

    返回值
    ----------
    共2列。第一列是k，第二列是sf。
    """
    from ovito.modifiers import CoordinationAnalysisModifier
    from ovito.io import import_file
    delta = cutoff/num_of_bins
    k_delta = k_cutoff/k_number_of_bins
    pipeline = import_file(filename)
    r = np.linspace(delta/2, cutoff-delta/2, num=num_of_bins)
    k = np.linspace(k_delta/2, k_cutoff - k_delta/2, num=k_number_of_bins)
    s_k = np.zeros(len(k))
    if keyword == "all":
        modifier = CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=num_of_bins)
        pipeline.modifiers.append(modifier)
        if end < 0:
            end += pipeline.source.num_frames+1
        for frame in range(end-(num_of_confs-1)*every, end+1, every):
            data = pipeline.compute(frame)
            current_rdf = data.series["coordination-rdf"].xy()
            gr = current_rdf[:, 1]
            rho = data.particles.count / data.cell.volume
            for i in range(len(k)):
                s_k[i] += 1+4*np.pi*rho*np.trapz(r*np.sin(k[i]*r)/k[i]*(gr-1), r)
    else:
        modifier = CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=num_of_bins, partial=True)
        pipeline.modifiers.append(modifier)
        if end < 0:
            end += pipeline.source.num_frames+1
        for frame in range(end-(num_of_confs-1)*every, end+1, every):
            data = pipeline.compute(frame)
            current_rdf = data.series["coordination-rdf"].xy()
            index = data.tables["coordination-rdf"].y.component_names.index(keyword)
            gr = current_rdf[:, index+1]
            rho = data.particles.count / data.cell.volume
            for i in range(len(k)):
                s_k[i] += 1+4*np.pi*rho*np.trapz(r*np.sin(k[i]*r)/k[i]*(gr-1), r)
    s_k /= num_of_confs
    return np.column_stack((k, s_k))


def count_num_of_type(atoms):
    """
    功能
    ----------
    统计ase的atoms对象(单帧)原子种类数

    参数
    ----------
    atoms: ASE中的atoms对象(单帧)

    返回值
    ----------
    无
    """
    chemical_symbols = atoms.get_chemical_symbols()
    chemical_symbols_list = list()
    for i in range(len(chemical_symbols)):
        if chemical_symbols[i] not in chemical_symbols_list:
            chemical_symbols_list.append(chemical_symbols[i])
    return len(chemical_symbols_list)


def count_num_of_type_atoms(atoms, ele):
    """
    功能
    ----------
    统计ase的atoms对象(单帧)各个原子种类的数量

    参数
    ----------
    atoms: ASE中的atoms对象(单帧)
    ele: 元素列表(元素或者数字)

    返回值
    ----------
    无
    """
    symbols = atoms.get_chemical_symbols()
    number = np.zeros(len(ele))
    ele_str = list()
    for i in range(len(ele)):
        if type(ele[i]) == str:
            ele_str.append(ele[i])
        if type(ele[i]) == int:
            ele_str.append(myconstant.periodic_table[ele[i]-1])
    for i in range(len(ele)):
        for j in symbols:
            if j == ele_str[i]:
                number[i] += 1
        print("number of %s: %d" % (ele[i], number[i]))


def lammps_calculater(atoms,ele,pot,lmp_dir,style="atomic"):
    """
    功能
    ----------
    LAMMPS计算器, 输入ASE的atoms, 计算势函数的能量, 力和误差

    参数
    ----------
    atoms: ASE中的atoms对象(list)
    ele: 元素列表
    pot: 势函数语句路径(相对/绝对)
    lmp_dir: LAMMPS可执行文件绝对路径
    style: LAMMPS data文件的style, 比如atomic, charge

    返回值
    ----------
    ASE中的atoms对象(list)
    """
    from ase.calculators.singlepoint import SinglePointCalculator as SPC
    symbol_list = ele
    pid = os.getpid()
    myfun.mkdir("%d" % pid)
    myio.atoms2dump(atoms, "%d/dump.atom" % pid, symbol_list, style=style)
    myio.atoms2data(atoms[0], "%d/data.txt" % pid, symbol_list, style=style)
    f_input = open("%d/in.rerun.txt" % pid, "w")
    f_input.write("units metal\n")
    f_input.write("dimension 3\n")
    f_input.write("boundary p p p\n")
    if style=="atomic":
        f_input.write("atom_style atomic\n")
    if style=="charge":
        f_input.write("atom_style charge\n")
    f_input.write("neighbor 2.0 bin\n")
    f_input.write("thermo 1\n")
    f_input.write("box tilt large\n")
    f_input.write("read_data data.txt\n")
    f_input.write("thermo_style custom step etotal vol pxx pyy pzz pxy pxz pyz \n")
    for i in range(len(symbol_list)):
        f_input.write("mass %d 1.0e-20\n" % (i+1))
    f_pot = open(pot)
    for i in f_pot.readlines():
        f_input.write(i)
    f_pot.close()
    f_input.write("\n")
    f_input.write("dump 1 all custom 1 dump.rerun.atom id type x y z fx fy fz\n")
    f_input.write("dump_modify 1 sort id\n")
    if style=="atomic":
        f_input.write("rerun dump.atom dump x y z\n")
    if style=="charge":
        f_input.write("rerun dump.atom dump q x y z\n")
    f_input.close()
    os.chdir("%d" % pid)
    print('DID WE MAKE IT HERE?')
    subp = sp.Popen("%s -in in.rerun.txt > /dev/null 2>&1" %lmp_dir,shell=True)
    print('WHAT ABOUT HERE?',"%s -in in.rerun.txt > /dev/null 2>&1" %lmp_dir)
    subp.wait()
    os.chdir("..")
    atoms_lammps = io.read("%d/dump.rerun.atom" % pid, ":", format="lammps-dump-text")
    df = myio.read_lammps_thermo("%d/log.lammps" % pid)
    energy = df["TotEng"].to_numpy()
    atoms_results = list()
    for i in range(len(atoms)):
        atoms_append = Atoms(symbols=atoms[i].get_chemical_symbols(), positions=atoms[i].get_positions(), cell=atoms[i].get_cell(), pbc=atoms[i].get_pbc())
        calc = SPC(atoms=atoms_append, energy=energy[i], forces=atoms_lammps[i].get_forces())
        atoms_append.set_calculator(calc)
        atoms_results.append(atoms_append)
    shutil.rmtree("%d" % pid)
    return atoms_results
