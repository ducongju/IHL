#!/bin/bash
#SBATCH --job-name=IHL
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

eval "$(/opt/app/anaconda3/bin/conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib

conda create -n IHL python=3.7 pytorch=1.10 cudatoolkit=11.1 torchvision -c pytorch -y
conda activate IHL

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

pip install -r requirements.txt
pip install -v -e .
