#!/bin/bash
module load gcc/11.2.0
module load cmake openmpi/4.0.5-intel-18.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/projects/qmchamm/shared/scott_builds/libtorch/lib"
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

export SCRIPTS_DIR=../scripts

while getopts :m:s:p:t:n:v: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        s) steps=${OPTARG};;
        p) pres="-p ${OPTARG}";;
        t) temp="-t ${OPTARG}";;
        v) velocity="-v ${OPTARG}";;
        n) nbeads="-n ${OPTARG}";;
    esac
done

echo "model: $model";

mkdir -p $MLFLOW_RUN_ID
cd $MLFLOW_RUN_ID

node=$(hostname)
node=${node%%.*}
cur_time=$(date +%Y%m%d%H%M%S)
IPI=$(which i-pi)
lmp=/projects/qmchamm/shared/scott_builds/lammps/build_misc/bin/lmp
if [ -f simulation.restart ]
then
	python $SCRIPTS_DIR/edit_xml.py -i ../input.restart.xml -o input.xml $temp -a ${node}${cur_time} $pres $nbeads
	python -u $IPI input.xml > log.ipi.txt &

	sleep 6
else
	python $SCRIPTS_DIR/edit_xml.py -i ../input.xml -o input.xml $velocity $temp -a ${node}${cur_time} $pres $nbeads
	python -u $IPI input.xml > log.ipi.txt &

	sleep 6
fi
