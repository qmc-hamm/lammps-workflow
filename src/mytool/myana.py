'''
笔记
----------
一些常用的分析
'''
import numpy as np

def num_non_diamond_defect(filename,frame=0):
    '''
    功能
    ----------
    返回LAMMPS的dump文件第frame帧非diamond结构缺陷原子数

    参数
    ----------
    filename: LAMMPS的dump文件
    frame: 帧数

    返回值
    ----------
    int: 第frame帧非diamond结构缺陷原子数
    '''
    from ovito.modifiers import IdentifyDiamondModifier
    from ovito.io import import_file
    dia=IdentifyDiamondModifier()
    pipeline=import_file(filename)
    pipeline.modifiers.append(dia)
    if frame=="all":
        num_defect=list()
        for i in range(pipeline.source.num_frames):
            data=pipeline.compute(i)
            num_defect.append(data.tables["structures"]["Count"][0]+\
                data.tables["structures"]["Count"][4]+\
                    data.tables["structures"]["Count"][5]+\
                        data.tables["structures"]["Count"][6])
    else:
        num_defect=-1
        if frame<0:
            frame+=pipeline.source.num_frames+1
        data=pipeline.compute(frame)
        num_defect=data.tables["structures"]["Count"][0]+\
            data.tables["structures"]["Count"][4]+\
                data.tables["structures"]["Count"][5]+\
                    data.tables["structures"]["Count"][6]
    return num_defect

def num_ws_interstitial(filename,reference,frame=0):
    '''
    功能
    ----------
    返回LAMMPS的dump文件第frame帧ws分析间隙原子数

    参数
    ----------
    filename: LAMMPS的dump文件
    reference: 参考的LAMMPS的dump文件
    frame: 帧数

    返回值
    ----------
    int: 第一帧ws分析间隙原子数
    '''
    from ovito.io import import_file
    from ovito.pipeline import FileSource
    from ovito.modifiers import WignerSeitzAnalysisModifier
    ws=WignerSeitzAnalysisModifier()
    ws.reference=FileSource()
    ws.reference.load(reference)
    pipeline = import_file(filename)
    pipeline.modifiers.append(ws)
    if frame<0:
        frame+=pipeline.source.num_frames+1
    data=pipeline.compute(frame)
    num_defect=data.attributes["WignerSeitz.vacancy_count"]
    return num_defect

def stru_diamond_defect(filename,frame=0):
    '''
    功能
    ----------
    返回LAMMPS的dump文件第frame帧diamond结构

    参数
    ----------
    filename: LAMMPS的dump文件
    frame: 帧数

    返回值
    ----------
    ASE中的atoms对象
    '''
    from ovito.modifiers import IdentifyDiamondModifier
    from ovito.io import import_file
    from ase import Atoms
    dia=IdentifyDiamondModifier()
    pipeline=import_file(filename)
    pipeline.modifiers.append(dia)
    if frame<0:
        frame+=pipeline.source.num_frames+1
    data=pipeline.compute(frame)
    positions=data.particles['Position']
    particle_types=data.particles['Structure Type']
    dia_index=np.r_[np.argwhere(particle_types==1),np.argwhere(particle_types==2),np.argwhere(particle_types==3)].flatten()
    return Atoms(positions=positions[dia_index],cell=data.cell[0:3,0:3],pbc=True)

def stru_non_diamond_defect(filename,frame=0):
    '''
    功能
    ----------
    返回LAMMPS的dump文件第frame帧非diamond结构缺陷原子结构

    参数
    ----------
    filename: LAMMPS的dump文件
    frame: 帧数

    返回值
    ----------
    ASE中的atoms对象
    '''
    from ovito.modifiers import IdentifyDiamondModifier
    from ovito.io import import_file
    from ase import Atoms
    dia=IdentifyDiamondModifier()
    pipeline=import_file(filename)
    pipeline.modifiers.append(dia)
    if frame<0:
        frame+=pipeline.source.num_frames+1
    data=pipeline.compute(frame)
    positions=data.particles['Position']
    particle_types=data.particles['Structure Type']
    others_index=np.r_[np.argwhere(particle_types==0),np.argwhere(particle_types==4),np.argwhere(particle_types==5),np.argwhere(particle_types==6)].flatten()
    return Atoms(positions=positions[others_index],cell=data.cell[0:3,0:3],pbc=True)
