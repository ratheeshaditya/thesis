#!/bin/bash
# parameters for slurm
#SBATCH -J training                   # job name, don't use spaces, keep it short
#SBATCH -c 32                          # number of cores, 1
#SBATCH --gres=gpu:1                  # number of gpus 1, some clusters don't have GPUs
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --mail-type=END,FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=a.m.joseph@student.utwente.nl   # Where to send mail to
#SBATCH --output=/home/s2435462/HRC/out/log/new/training_%j.log      # Standard output and error log
#SBATCH --error=/home/s2435462/HRC/out/err/new/training_%j.err                # if yoou want the errors logged seperately
#SBATCH --partition=main # Here 50..is the partition name..can be checked via sinfo


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

# python -u ../train_transformer_cross_val_NTU.py --filename training_NTU_2D_128d_100e_10p_001 --lr 0.001 --embed_dim 128 --dataset NTU_2D --model_type temporal --epochs 100 --patience 10
# python -u ../train_transformer_cross_val_NTU.py --config_file ../config_debug.yml
# python -u ../train_transformer_cross_val_NTU.py --filename temporal_2 --lr 0.001 --embed_dim 128 --dataset NTU_2D --model_type temporal_2 --epochs 100 --patience 10 --batch_size 2000
python -u ../train_transformer_cross_val_NTU.py --config_file ../config.yml
