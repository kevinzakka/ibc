#!/usr/bin/env bash

set -e
set -x

for train_size in 10
do
    EXPERIMENT_NAME=explicit_mse_${train_size}

    python train.py \
        --experiment-name $EXPERIMENT_NAME \
        --train-dataset-size $train_size \
        --policy-type EXPLICIT \
        --dropout-prob 0.2 \
        --weight-decay 1e-3 \
        --max-epochs 6000 \
        --learning-rate 1e-3 \
        --train-batch-size 128 \
        --spatial-reduction SPATIAL_SOFTMAX \

    python plot.py --experiment-name $EXPERIMENT_NAME
done
