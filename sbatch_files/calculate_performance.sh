#!/bin/bash
# parameters for slurm
#SBATCH -J calculate_performance                   # job name, don't use spaces, keep it short
#SBATCH -c 2                          # number of cores, 1
#SBATCH --gres=gpu:1                  # number of gpus 1, some clusters don't have GPUs
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --mail-type=END,FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=a.m.joseph@student.utwente.nl   # Where to send mail to
#SBATCH --output=/home/s2435462/HRC/out/log/calculate_performance_%j.log      # Standard output and error log
#SBATCH --error=/home/s2435462/HRC/out/err/calculate_performance_%j.err                # if yoou want the errors logged seperately
#SBATCH --partition=main # Here 50..is the partition name..can be checked via sinfo
 
# Create a directory for this job on the node
echo $PWD
 
# load all modules needed 
module load nvidia/cuda-10.2

 
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

python ../calculate_performance.py --filename tensorboard_test_7