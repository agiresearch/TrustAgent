#!/bin/bash
#SBATCH --partition default_queue # the job will be submitted to the "default_queue" partition. Running the command ==sinfo== will show you the various queues, and the worker nodes allocated to them.
##SBATCH --ntasks=1 # run four tasks (usually you will want a single task unless you run mpi jobs.)
##SBATCH --nodes=1-2 # request that the tasks be assigned on a minimum of 1 node and a maximum of 2 nodes.
#SBATCH --cpus-per-task=24 # each task will be assigned 2 cpus. All cpus will be allocated on the same machine (useful for multithread jobs).
#SBATCH --time=6-12:00:00 # request a running time of 1 day and 12 hours. If the job exceeds this time it is killed. The default is different for each partition (see above)
##SBATCH --mem=320G # request 16G or memory per node. Alternative to --mem-per-cpu option. --mem and --mem-per-cpu options are mutually exclusive.
#SBATCH --job-name=llama2_13b # the name of the job. Default is the name of the script file
#SBATCH --output=llama2_13b_%j.out # file to which stdout will be written (%j is replaced with the job id). Default slurm-%j.out
#SBATCH --error=llama2_13b_%j.err # file to which stderr will be written (%j is replaced with the job id). Default same as output
##SBATCH --constraint="IB&Haswell" # request that the job be ran on a node that has infiniband and haswell processors. Default no constraints.
#SBATCH --mail-type=END,FAIL # send email when job finishes or when it fails
#SBATCH --mail-user=xianjunyang@ucsb.edu
#SBATCH --nodelist=cipr-gpu08 # request that the job runs on a certain node
#SBATCH --exclude=cipr-gpu03,cipr-gpu04 # exclude these nodes from the resources granted to the job. Job will not be scheduled on any of these nodes.
#SBATCH --reservation=dsss_a100
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=20G

source /mnt/dsss_data/xianjun/anaconda3/etc/profile.d/conda.sh  # Assuming this is the correct path to conda.sh in your Anaconda installation
conda activate vllm
# model_name_or_path /home/dsi/xyang/project_data/Llama-2-13b-hf
torchrun --nproc_per_node=4 main.py