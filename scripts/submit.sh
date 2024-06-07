for p in $(seq 50 25 200)
do
	for case in QMC
	do
		for temp in $(seq 300 100 300)
		do
			mkdir -p p$p/cache/$case/$temp/analysis
			cp p$p/data.txt p$p/cache/$case/$temp
			cp p$p/init.xyz p$p/cache/$case/$temp
			cd p$p/cache/$case/$temp
			nbeads=16
			ntasks=1
			JOB=$(sbatch \
			-p secondary \
			-N 1 \
		        --ntasks=$ntasks \
			--exclusive \
			-t 03:58:00 \
			--export=ALL,temp=$temp,case=$case,pres=$p,ntasks=$ntasks,nbeads=$nbeads \
			-J PIMD_${p}_${temp}_${nbeads}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			../../../../slurm.sh | tr -cd "[0-9]")
			for j in $(seq 1 1 12)
			do
				JOB=$(sbatch \
				-p secondary \
				-N 1 \
				--ntasks=$ntasks \
				-t 03:58:00 \
				--exclusive \
				--export=ALL,temp=$temp,case=$case,pres=$p,ntasks=$ntasks,nbeads=$nbeads \
				-J PIMD_${p}_${temp}_${nbeads}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../../slurm.sh | tr -cd "[0-9]")
			done
			cd ../../../..
		done
	done
done
