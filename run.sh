#!/bin/bash

# Quit if there are any errors
set -e

data_dir="./data"
read -p "Choose the dataset (1: twitter, 2: stress): " dataset
read -p "Choose the experiment (1: base, 2: simple, 3: label, 4: fact, 5: fact_human): " experiment

case $dataset in
  1)
    data_dir="${data_dir}/twitter"
    ;;
  2)
    data_dir="${data_dir}/stress"
    ;;
  *)
    echo "Invalid dataset"
    exit 1
    ;;
esac

case $experiment in
  1)
    experiment="base"
    ;;
  2) 
    experiment="simple"
    ;;
  3) 
    experiment="label"
    ;;
  4) 
    experiment="fact"
    ;;
  5) 
    experiment="fact_human"
    ;;
  *)
    echo "Invalid experiment"
    exit 1
    ;;
esac


lr=0.00005
batch_size=16
n_epochs=20
warmup_ratio=0.1

seed=42


CUDA_VISIBLE_DEVICES=$1 python3 run.py \
  --data_dir $data_dir \
  --experiment $experiment \
  --lr $lr \
  --batch_size $batch_size \
  --n_epochs $n_epochs \
  --warmup_ratio $warmup_ratio \
  --seed $seed
