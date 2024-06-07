"""
笔记
----------
一些常用的画图函数
"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from mytool import myio
from mytool import myfun
from mytool import myconstant


def myfont(font_type):
    return FontProperties(fname="%s/fonts/%s.ttf"%(myconstant.path2mysharelib,font_type))


def plot_lammps_thermo(f_thermo,f_png,title=None,tmax=None,tmin=None):
    """
    功能
    ----------
    根据lammps的thermo文件画图

    参数
    ----------
    f_thermo:lammps的thermo文件
    f_png:输出的png文件
    title(可选):图的标题
    tmax(可选):图中最大温度
    tmin(可选):图中最小温度

    返回值
    ----------
    无

    其他
    ----------
    推荐LAMMPS命令：thermo_style custom step time temp pe ke etotal vol density lx ly lz xy xz yz press pxx pyy pzz pxy pxz pyz
    """
    df=myio.read_lammps_thermo(f_thermo)
    fig,axes=plt.subplots(nrows=7,ncols=1,figsize=(7,7),sharex=True)
    temperature=df["Temp"]
    step=df["Step"]
    energy=df["TotEng"]
    volume=df["Volume"]
    stress=df[["Pxx","Pyy","Pzz","Pxy","Pxz","Pyz"]]/10000
    cell=df[["Lx","Ly","Lz","Xy","Xz","Yz"]]
    axes[0].plot(step,energy)
    axes[0].set_ylabel(r"E(eV)")
    axes[0].set_xticks([])
    axes[1].plot(step,temperature)
    axes[1].set_ylabel(r"T(K)")
    axes[1].set_xticks([])
    if tmax!=None and tmin!=None:
        axes[1].set_ylim(tmin,tmax)
    axes[2].plot(step,volume)
    axes[2].set_ylabel(r"V($\AA^3$)")
    axes[2].set_xticks([])
    axes[3].plot(step,cell["Lx"],label="xx")
    axes[3].plot(step,cell["Ly"],label="yy")
    axes[3].plot(step,cell["Lz"],label="zz")
    axes[3].set_ylabel(r"$Cell_d$($\AA$)")
    axes[3].set_xticks([])
    axes[3].legend()
    axes[4].plot(step,cell["Xy"],label="xy")
    axes[4].plot(step,cell["Xz"],label="xz")
    axes[4].plot(step,cell["Yz"],label="yz")
    axes[4].set_ylabel(r"$Cell_{od}$($\AA$)")
    axes[4].set_xticks([])
    axes[4].legend()
    axes[5].plot(step,stress["Pxx"],label="xx")
    axes[5].plot(step,stress["Pyy"],label="yy")
    axes[5].plot(step,stress["Pzz"],label="zz")
    axes[5].set_ylabel(r"$P_d$(GPa)")
    axes[5].set_xticks([])
    axes[5].legend()
    axes[6].plot(step,stress["Pxy"],label="xy")
    axes[6].plot(step,stress["Pxz"],label="xz")
    axes[6].plot(step,stress["Pyz"],label="yz")
    axes[6].set_ylabel(r"$P_{od}$(GPa)")
    axes[6].set_xticks(myfun.seq(min(step),(max(step)-min(step))//10,max(step)))
    axes[6].legend()
    axes[-1].set_xlabel(r"Step")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(fname=f_png,format="png",dpi=320)
    plt.close("all")


def plot_lcurve(f_lcurve,f_png,title=None,log_scale=True,virial=False,emax=1E2,emin=1E0,fmax=1E3,fmin=1E1,vmax=1E2,vmin=1E1,smooth=False,window_length=51,polyorder=3):
    """
    功能
    ----------
    根据deepmd的lcurve.out画图

    参数
    ----------
    f_lcurve:deepmd的lcurve.out
    f_png:输出的png文件
    title(可选):图的标题,默认值为None
    log_scale(可选):是否为对数坐标,默认值为True
    virial:是否画virial信息,默认值为False
    emax(可选):能量误差最大值
    emin(可选):能量误差最小值
    fmax(可选):力误差最大值
    fmin(可选):力误差最小值
    vmax(可选):virial误差最大值
    vmin(可选):virial误差最小值
    smooth(可选):是否平滑曲线,默认值为False
    window_length(可选):平滑曲线参数
    polyorder(可选):平滑曲线参数

    返回值
    ----------
    无
    """
    data_frame=pd.read_csv(f_lcurve,sep=r"\s+",names=open(f_lcurve,"r").readline().split()[1:],skiprows=1)
    column=[["l2_e_tst","l2_e_trn"],["l2_f_tst","l2_f_trn"],["l2_v_tst","l2_v_trn"]]
    if smooth==True:
        for i in range(2+(virial==True)):
            for j in range(2):
                data_frame[column[i][j]]=savgol_filter(data_frame[column[i][j]],window_length,polyorder)
    fig,axes=plt.subplots(2+(virial==True),1,sharex=True)
    for i in range(2+(virial==True)):
        axes[i].plot(pd.to_numeric(data_frame["batch"],errors="coerce"),pd.to_numeric(data_frame[column[i][0]],errors="coerce")*1000,label="training",alpha=0.5)
        axes[i].plot(pd.to_numeric(data_frame["batch"],errors="coerce"),pd.to_numeric(data_frame[column[i][1]],errors="coerce")*1000,label="test",alpha=0.5)
        axes[i].legend()
    axes[0].set_ylim(emin,emax)
    axes[1].set_ylim(fmin,fmax)
    axes[0].set_ylabel(r"E RMSE (meV/atom)")
    axes[1].set_ylabel(r"F RMSE (meV/$\AA$)")
    if virial==True:
        axes[2].set_ylim(vmin,vmax)
        axes[2].set_ylabel(r"V RMSE (meV/$\AA$)")
    axes[-1].set_xlabel(r"Training batch")
    if title:
        axes[0].set_title(title)
    if log_scale:
        for i in range(2+(virial==True)):
            axes[i].set_xscale("log")
            axes[i].set_yscale("log")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(fname=f_png,format="png",dpi=320)
    plt.close("all")


def plot_vasp_aimd_outcar(f_outcar,f_png,title=None,tmax=None,tmin=None):
    """
    功能
    ----------
    根据vasp的aimd计算的OUTCAR文件画图

    参数
    ----------
    f_outcar:vasp的aimd计算的OUTCAR文件
    f_png:输出的png文件
    title(可选):图的标题
    tmax(可选):图中最大温度
    tmin(可选):图中最小温度

    返回值
    ----------
    无
    """
    atoms=myio.outcar2atoms(f_outcar)
    temperature=myio.read_vasp_temperature(f_outcar)
    fig,axes=plt.subplots(nrows=7,ncols=1,figsize=(7,7),sharex=True)
    step=myfun.seq(len(atoms))
    energy=np.array([atoms[i].get_potential_energy() for i in range(len(atoms))])
    energy/=len(atoms[0].get_positions())
    volume=[atoms[i].get_volume() for i in range(len(atoms))]
    stress=np.zeros((len(atoms),6))
    for i in range(6):
        stress[:,i]=[atoms[j].get_stress()[i] for j in range(len(atoms))]
    cell=np.zeros((len(atoms),6))
    for i in range(3):
        cell[:,i]=[atoms[j].get_cell()[i][i] for j in range(len(atoms))]
    cell[:,3]=[atoms[j].get_cell()[0][1] for j in range(len(atoms))]
    cell[:,4]=[atoms[j].get_cell()[0][2] for j in range(len(atoms))]
    cell[:,5]=[atoms[j].get_cell()[1][2] for j in range(len(atoms))]
    axes[0].plot(step,energy)
    axes[0].set_ylabel(r"E(eV/atom)")
    axes[0].set_xticks([])
    axes[1].plot(step,temperature)
    axes[1].set_ylabel(r"T(K)")
    axes[1].set_xticks([])
    if tmax!=None and tmin!=None:
        axes[1].set_ylim(tmin,tmax)
    axes[2].plot(step,volume)
    axes[2].set_ylabel(r"V($\AA^3$)")
    axes[2].set_xticks([])
    axes[3].plot(step,cell[:,0],label="xx")
    axes[3].plot(step,cell[:,1],label="yy")
    axes[3].plot(step,cell[:,2],label="zz")
    axes[3].set_ylabel(r"$Cell_d$($\AA$)")
    axes[3].set_xticks([])
    axes[3].legend()
    axes[4].plot(step,cell[:,3],label="xy")
    axes[4].plot(step,cell[:,4],label="xz")
    axes[4].plot(step,cell[:,5],label="yz")
    axes[4].set_ylabel(r"$Cell_{od}$($\AA$)")
    axes[4].set_xticks([])
    axes[4].legend()
    axes[5].plot(step,stress[:,0],label="xx")
    axes[5].plot(step,stress[:,1],label="yy")
    axes[5].plot(step,stress[:,2],label="zz")
    axes[5].set_ylabel(r"$P_d$(GPa)")
    axes[5].set_xticks([])
    axes[5].legend()
    axes[6].plot(step,stress[:,3],label="yz")
    axes[6].plot(step,stress[:,4],label="xz")
    axes[6].plot(step,stress[:,5],label="xy")
    axes[6].set_ylabel(r"$P_{od}$(GPa)")
    axes[6].set_xticks(myfun.seq(0,len(step)//10,len(step)))
    axes[6].legend()
    axes[-1].set_xlabel(r"Step")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(fname=f_png,format="png",dpi=320)
    plt.close("all")
