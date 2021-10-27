#!/usr/bin/env bash

set -e
set -x

for train_size in 10 30
do
    EXPERIMENT_NAME=explicit_mse_${train_size}

    python train.py \
        --experiment-name $EXPERIMENT_NAME \
        --train-dataset-size $train_size \
        --policy-type EXPLICIT \
        --dropout-prob 0.1 \
        --weight-decay 1e-4 \
        --max-epochs 2000 \
        --learning-rate 1e-3 \
        --spatial-reduction SPATIAL_SOFTMAX \
        --coord-conv \

    python plot.py --experiment-name $EXPERIMENT_NAME
done
