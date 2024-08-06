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

export SCRIPTS_DIR=$(realpath ../scripts)

while getopts :m:s:p:t:n:v: flag
do
    case "${flag}" in
        s) steps=${OPTARG};;
        p) presure_option="-p ${OPTARG}";pressure=${OPTARG};;
        t) temp=${OPTARG};;
        v) velocity="-v ${OPTARG}";;
        n) nbeads="-n ${OPTARG}";;
        c) case=${OPTARG};;
    esac
done
pot_dir=$(realpath .)
cd p$pressure
echo "cd into $(realpath .)"

mkdir -p cache/$case/$temp/analysis
cp data.txt cache/$case/$temp
cp init.xyz cache/$case/$temp
cd cache/$case/$temp

if [ -f simulation.restart ]
then
	python $SCRIPTS_DIR/edit_xml.py -i ../../../../input.restart.xml -o input.xml -t $temp -a ${node}${cur_time} $presure_option $nbeads
	python -u $IPI input.xml > log.ipi.txt &

	sleep 60

	$lmp -v address ${node}${cur_time} \
	-v pot_dir $pot_dir \
	-in ../in.mace.$case.txt &
else
	python $SCRIPTS_DIR/edit_xml.py -i ../../../../input.xml -o input.xml $velocity -t $temp -a ${node}${cur_time} $presure_option $nbeads
	python -u $IPI input.xml > log.ipi.txt &

	sleep 60
	$lmp -v address ${node}${cur_time} \
	-v pot_dir $pot_dir \
	-in ../in.mace.$case.txt &
fi
