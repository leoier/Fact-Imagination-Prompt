# Large Language Model for Generating Twitter Data in Sentiment Analysis

This repo contains the code and data for finetuning the sentiment classification model used in *Large Language Model for Generating Twitter Data in Sentiment Analysis*.

## Environment Setup
The code is built with Python 3.10.
Other package requirements are listed in `requirements.txt`.
You are suggested to run the code in an isolated virtual [conda](https://www.anaconda.com/) environment.
Suppose you have already install conda in your device, you can create a new environment and activate it with
```bash
conda create -n 310 python=3.10
conda activate 310
```
Then, you can install the required packages with
```bash
pip install -r requirements.txt
```

Alternatively, you can also use other Python version manager or virtual environments such as [pyenv](https://github.com/pyenv/pyenv) or [docker](https://www.docker.com/) to you prefer.


## Data
The data used in this project are processed and stored in the [`data`](./data) folder.


## Run

If you are using a Unix-like system such as Linux or MacOS, you can run the code through the provided [`run.sh`](./run.sh) file. And then following the instructions to run the experiemnt on the datasets.
```bash
./run.sh [GPU ID]
```
for example, 
```bash
./run.sh 0
```
if you want to GPU-0 to accelerate your training.
If you leave `GPU ID` blank, the model will be trained on CPU.


Alternately, you can also run the code with 
```bash
[CUDA_VISIBLE_DEVICES=...] python3 run.py \
      --data_dir "${data_dir}/${dataset}" \
      --experiment ${experiment} \
      --lr ${lr} \
      --batch_size ${batch_size} \
      --n_epochs ${n_epochs} \
      --warmup_ratio ${warmup_ratio} \
      --seed ${seed} \
      --log_path "log/${dataset}_${experiment}.log" \
      --output_path "output/${dataset}_${experiment}.csv"
```

The output will be saved in `output/${dataset}_${experiment}.csv` and the log will be saved in `log/${dataset}_${experiment}.log`.