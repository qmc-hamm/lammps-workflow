#!/bin/bash
module load cmake openmpi
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/projects/bcqo/shubhanggoswami/scott_builds/libtorch/lib"
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
node=$(hostname)
node=${node%%.*}
cur_time=$(date +%Y%m%d%H%M%S)
IPI=$(which i-pi)
lmp=/projects/bcqo/shubhanggoswami/lammps/build/bin/lmp

export SCRIPTS_DIR=../../scripts

while getopts :m:s:p:t:n:v: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        s) steps=${OPTARG};;
        p) presure_option="-p ${OPTARG}";pressure=${OPTARG};;
        t) temp="-t ${OPTARG}";;
        v) velocity="-v ${OPTARG}";;
        n) nbeads="-n ${OPTARG}";;
        c) case=${OPTARG};;
    esac
done

echo "model: $model";
cd p$pressure
echo "Cd into $(realpath .)"
node=$(hostname)
node=${node%%.*}
cur_time=$(date +%Y%m%d%H%M%S)
IPI=$(which i-pi)
lmp=/projects/qmchamm/shared/scott_builds/lammps/build_misc/bin/lmp
if [ -f simulation.restart ]
then
	python $SCRIPTS_DIR/edit_xml.py -i ../../input.restart.xml -o input.xml $temp -a ${node}${cur_time} $presure_option $nbeads
	python -u $IPI input.xml > log.ipi.txt &

	sleep 6
else
	python $SCRIPTS_DIR/edit_xml.py -i ../../input.xml -o input.xml $velocity $temp -a ${node}${cur_time} $presure_option $nbeads
	python -u $IPI input.xml > log.ipi.txt &

	sleep 60
	$lmp -v address ${node}${cur_time} \
	-v pot_dir $(realpath ..) \
	-in ../in.mace.$case.txt &
fi
