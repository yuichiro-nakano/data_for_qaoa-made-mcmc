#!/bin/bash
#PBS -l select=1:ncpus=8
#PBS -N samplejob

source ~/.bashrc
cd ${PBS_O_WORKDIR}
cd src
conda activate quantum2
python spectral_gap.py