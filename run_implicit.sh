#!/usr/bin/env bash

set -e
set -x

for train_size in 10 30
do
    EXPERIMENT_NAME=implicit_ebm_${train_size}

    python train.py \
        --experiment-name $EXPERIMENT_NAME \
        --train-dataset-size $train_size \
        --policy-type IMPLICIT \
        --dropout-prob 0. \
        --weight-decay 0. \
        --max-epochs 3000 \
        --learning-rate 1e-3 \
        --spatial-reduction SPATIAL_SOFTMAX \

    python plot.py --experiment-name $EXPERIMENT_NAME
done
