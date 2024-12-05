#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a10g-8-gm192-c192-m768
bash
conda init bash >/dev/null 2>&1
source ~/.bashrc
cd /scratch/ycai222/Adversarial-Attack-Analysis-on-CNN-Image-Classification
conda activate advattack
 
echo '====start running===='
python transformer.py >> ./logs/epch10_ViT_finetune.log 2>> ./logs/epch10_ViT_finetune.err
echo '=====end======='