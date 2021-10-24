#!/usr/bin/env bash

set -e
set -x

for train_size in 10 30
do
    EXPERIMENT_NAME=explicit_mse_${train_size}

    python train.py \
        --experiment-name $EXPERIMENT_NAME \
        --train-dataset-size $train_size \
        --policy-type EXPLICIT_MSE \
        --coord-conv \
        --dropout-prob 0.2 \
        --weight-decay 3e-4 \
        --max-epochs 500 \
        --spatial-reduction SPATIAL_SOFTMAX \

    python plot.py --experiment-name $EXPERIMENT_NAME
done
