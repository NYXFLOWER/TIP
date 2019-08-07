#!/bin/bash
echo "Hello from ${JOB_ID}. "

# name
#$ -N TrainDecagon

# real memory
#$ -l rmem=32G

# core
#$ -pe openmp 1

# time
#$ -l h_rt=96:00:00

# mail
#$ -M hxu31@sheffield.ac.uk
# Email notifications if the job aborts
#$ -m a

# Setup env
export PATH="~/anaconda3/bin:$PATH"
module load apps/python/conda
module load libs/CUDA/10.0.130/binary
source activate hhh


# run
