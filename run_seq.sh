#!/bin/bash
#SBATCH --job-name=spur_base
#SBATCH -t 6-00:00                    # tiempo maximo en el cluster (D-HH:MM)
#SBATCH --gres=gpu
#SBATCH -o investigacion/spur/exp_logs/baseline_%j.out                 # STDOUT (A = )
#SBATCH -e investigacion/spur/exp_logs/baseline_%j.err                 # STDERR
#SBATCH --mail-type=END,FAIL         # notificacion cuando el trabajo termine o falle
#SBATCH --mail-user=afraymon@uc.cl   # mail donde mandar las notificaciones
#SBATCH --workdir=/user/araymond/storage     # direccion del directorio de trabajo
#SBATCH --partition=ialab-high
#SBATCH --nodelist hydra             # forzamos scylla
#SBATCH --nodes 1                    # numero de nodos a usar
#SBATCH --ntasks-per-node=1          # numero de trabajos (procesos) por nodo
#SBATCH --cpus-per-task=1            # numero de cpus (threads) por trabajo (proceso)
#SBATCH --mem=80G

source p3/bin/activate
cd investigacion/spur

# for s in 222
# do
#	for asc in 0 1
#	do
#		for p in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#		do
#			for env in nobg gradient images
#			do
#				HDF5_USE_FILE_LOCKING=FALSE python main.py --seed $s --forget_asc $asc --forget_t $p --env $env
#			done
#		done
#	done
# done
# echo "Finished with job $SLURM_JOBID"

				
