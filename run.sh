#!/bin/bash

# Quit if there are any errors
set -e

data_dir="./data"

lr=0.00005
batch_size=16
n_epochs=20
warmup_ratio=0.1

seed=42


read -p "Choose the dataset (0: all, 1: twitter, 2: covid-19): " dataset_choice
read -p "Choose the experiment (0: all, 1: base, 2: simple, 3: label, 4: fact, 5: fact_human): " experiment_choice

case $dataset_choice in
  0) 
    datasets=("twitter" "covid-19")
    ;;
  1)
    datasets=("twitter")
    ;;
  2)
    datasets=("covid-19")
    ;;
  *)
    echo "Invalid dataset"
    exit 1
    ;;
esac


case $experiment_choice in
  0) 
    experiments=("base" "simple" "label" "fact" "fact_human")
    ;;
  1)
    experiments=("base")
    ;;
  2) 
    experiments=("simple")
    ;;
  3) 
    experiments=("label")
    ;;
  4) 
    experiments=("fact")
    ;;
  5) 
    experiments=("fact_human")
    ;;
  *)
    echo "Invalid experiment"
    exit 1
    ;;
esac

for dataset in "${datasets[@]}"; do
  for experiment in "${experiments[@]}"; do
    echo "Running ${experiment} on ${dataset}..."
    CUDA_VISIBLE_DEVICES=$1 python3 run.py \
      --data_dir "${data_dir}/${dataset}" \
      --experiment $experiment \
      --lr $lr \
      --batch_size $batch_size \
      --n_epochs $n_epochs \
      --warmup_ratio $warmup_ratio \
      --seed $seed \
      --log_path "log/${dataset}_${experiment}.log" \
      --output_path "output/${dataset}_${experiment}.csv"
  done
done
