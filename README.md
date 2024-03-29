# Layer-Masking

Running Command:

```
torchrun --nproc_per_node=1 train.py --model vit_base --epochs 30 --batch-size 64 --opt adamw --lr 0.003 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 10 --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --tuning_method tune_attention
```

Running Command for GD Mask Learning:

```
python train_mask_GD.py --model vit_base --epochs 50 --batch-size 128 --opt adamw --lr 1e-4 --outer_lr 1e-3 --lr_scaler 100 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 15 --lr-warmup-decay 0.033 --tuning_method tune_attention_blocks_random --dataset breastUS --wandb_logging
```

Visaulizing Optuna Runs on a Web Browser


```
optuna-dashboard sqlite:///example-study.db
ssh -L 8888:localhost:8080 co-dutt1@login-gpu.hpc.cam.ac.uk
```
