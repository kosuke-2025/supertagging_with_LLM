#!/bin/sh
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=gr22
#PBS -j oe
#PBS -o train.out

cd /work/gr22/r22001/workspace/supertagging_with_LLM/

module load python/3.10.16 
module load cuda/12.9
module unload nvidia/25.9

source .venv/bin/activate

python train.py

deactivate