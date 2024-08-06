from ase import io
import sys
from qharv_db.ase_md import mhcpc_supercell
from mytool import myio
from mytool import myfun

def main():
    if len(sys.argv) != 2:
       print("Usage: python gen_xyz.py [pressure]")
       sys.exit(1)

    pgpa=int(sys.argv[1])
    natom=576
    atoms=mhcpc_supercell(pgpa,natom)
    myfun.mkdir("p{}".format(pgpa))
    myio.atoms2data(atoms,"p{}/data.txt".format(pgpa),["H"])
    myio.atoms2ipixyz(atoms,"p{}/init.xyz".format(pgpa))

if __name__ == '__main__':
    main()
