#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account BMAI-CDT-SL2-GPU
#SBATCH --time=36:00:00

cd /home/co-dutt1/rds/hpc-work/Layer-Masking
module load cuda/11.2
module load cudnn/8.1_cuda-11.2
source ~/rds/hpc-work/miniconda3/bin/activate pytorch

model="vit_base"
epochs=30
warmup_epochs=29
batch_size=512
lr=1e-4
tuning_method="tune_attention_blocks_random"
dataset="HAM10000"

for i in {1..50}
do
   python train.py --model $model --epochs $epochs --batch-size $batch_size --opt adamw --lr $lr --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs $warmup_epochs --lr-warmup-decay 0.033 --amp --tuning_method $tuning_method --dataset $dataset
   echo "FINISHED ITERATION $i"
   echo
done
