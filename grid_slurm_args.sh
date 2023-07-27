#!/bin/bash
#
#============== SLURM SETUP ================#
#SBATCH --cpus-per-task=14
#SBATCH -o %x.%j.%N.out
#SBATCH -e %x.%j.%N.err
#SBATCH -J MOR
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBatch --partition=cm2_std
#SBATCH --qos=cm2_std
#SBATCH --nodes=12
#SBATCH --tasks-per-node=2
#SBATCH --mail-type=all
#SBATCH --mail-user=benjamin.oelke@tum.de
#SBATCH --export=ALL
#SBATCH --time=01:15:00

module load slurm_setup
source load_jobfarm.sh
#bash jobfarm_test.sh
jobfarm start "/dss/dsshome1/lxc0F/ga73jiv2/Mordor/${1}"

exit 0
