# Layer-Masking

Running Command:

```
torchrun --nproc_per_node=1 train.py --model vit_base --epochs 30 --batch-size 64 --opt adamw --lr 0.003 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 10 --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --tuning_method tune_attention
```