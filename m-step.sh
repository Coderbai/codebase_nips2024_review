#! /bin/bash
#SBATCH -o log/output.%j.out
#SBATCH --partition=GPUA100
#SBATCH --job-name=ciden
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --qos=normal
#SBATCH --cpus-per-task=5
#SBATCH --time 100:00:00
#SBATCH --mem 24G
#SBATCH -x gpua800n23,gpua800n02,gpua800n15

source /gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh
conda activate DPS

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u -m accelerate.commands.launch denoiser_diffusion.py --exp training --dataset_path path_to_dataset --model_path path_to_checkpoint --ema


