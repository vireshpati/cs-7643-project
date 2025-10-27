# Irregular Sampling-Aware Positional Encoding for Time Series Transformers

## Environment

``` 
pip install -r ./requirements.txt
```

Default CUDA device is 0 and can be changed by setting the `CUDA_VISIBLE_DEVICES` environment variable in `./Time-Series-Library/scripts/**/*.sh`. 

Default CUDA version is 12.6 and can be changed in `./requirements.txt`.

## Datasets

We evaluate our model on the following long-term forecasting datasets:
- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Electricity
- Traffic
- Weather
- Exchange

Download the datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) and place them in `./Time-Series-Library/dataset`

### Metrics 

MSE, MAE 

### Series Lengths

96, 192, 384, 720 for all datasets

## Weights and Biases

Use wandb to track the training and evaluation progress.

```
export WANDB_API_KEY=wandb_api_key_here
```

or

```
wandb login
```

## Example Usage

```
conda create -n tslib python=3.10
conda activate tslib
pip install -r ./requirements.txt
cd ./Time-Series-Library/
bash ./scripts/long_term_forecast/ETT_script/Transformer_ETTh1_grid.sh
```