#!/bin/bash
# parameters for slurm
#SBATCH -J training                   # job name, don't use spaces, keep it short
#SBATCH -c 32                          # number of cores, 1
#SBATCH --gres=gpu:1                  # number of gpus 1, some clusters don't have GPUs
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --mail-type=END,FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --output=/home/s2765918/code-et/out/training_%j.log      # Standard output and error log
#SBATCH --error=/home/s2765918/code-et/out/training_%j.err                # if yoou want the errors logged seperately
#SBATCH --partition=dmb # Here 50..is the partition name..can be checked via sinfo


# Maximum that worked - c:32, mem:64, gpu:1 
# Create a directory for this job on the node
echo $PWD
 
# load all modules needed 
# module load nvidia/cuda-10.2
module load nvidia/cuda-11.3
 
# It's nice to have some information logged for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : "$SLURM_NNODES
echo "Number of MPI ranks         : "$SLURM_NTASKS
echo "Number of threads           : "$SLURM_CPUS_PER_TASK
echo "Number of MPI ranks per node: "$SLURM_TASKS_PER_NODE
echo "Number of threads per core  : "$SLURM_THREADS_PER_CORE
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "
echo $CONDA_DEFAULT_ENV




python -u ../extract_features.py
