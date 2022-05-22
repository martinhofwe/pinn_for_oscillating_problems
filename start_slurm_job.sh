#!/bin/bash
#SBATCH --job-name="pinn job"
#SBATCH --workdir=/clusterFS/home/student/mhofwe/pinn/
# append output if file already exists
#SBATCH --open-mode=append
# number cpus per task
#SBATCH --cpus-per-task=1
# maximum time limit
# --time=DD-HH
# --time=HH:MM::SS
#SBATCH --time=02-00
# maximum memory limit
#SBATCH --mem=24G
# requires a gpu (otherwise: remove line)
#SBATCH --gres=gpu
# partitions to choose the nodes from (find e.g. using sview command)
# example for cpu only:
# SBATCH --partition=client,labor,short,simulation22
# example for gpu:
#SBATCH --partition=gpu,gpu2,gpu6
# if you are in the igpu partition (you will know when you are!)
# use the next three lines for running on ipgu. always exluce all hosts that are not yours or you have the okay of the owner to use that host!!
# SBATCH --partition=igpu
# SBATCH --exclude=sanderling,fritzfantom,ludwig,blackskimmer,calculus,anubis,ravenriad
# SBATCH --qos=igpu
# SBATCH --partition=gpu
# this can start multiple instances of the script with different array ids (env variable)
# one instance: --array=0
# three instances: --array=0-2
#SBATCH --array=0-15

USER = "mhofwe"
export HOME="/clusterFS/home/student/"$USER
export PYTHONPATH="/clusterFS/home/student/${USER}/pinn"

# catch maximum error code of the python script
error=0; trap 'error=$(($?>$error?$?:$error))' ERR

/clusterFS/home/student/mhofwe/miniconda3/envs/pinn_gpu/bin/python 'pinn.py'

exit $error
