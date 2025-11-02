#!/usr/bin/env bash

# Run Transformer with relative positional encoding across all long-term
# forecasting datasets using the shared grid.

set -euo pipefail

model_name=Transformer
pred_lens=(96 192 336 720)
enc_layers=(2)
dec_layers=(1)
positional_flag=(--positional_encoding relative)

run_transformer_grid() {
  local model_prefix=$1
  local data_flag=$2
  local root_path=$3
  local data_path=$4
  local features=$5
  local seq_len=$6
  local label_len=$7
  local enc_in=$8
  local dec_in=$9
  local c_out=${10}
  shift 10
  local extra_args=("$@")

  for pred_len in "${pred_lens[@]}"; do
    for e_layers in "${enc_layers[@]}"; do
      for d_layers in "${dec_layers[@]}"; do
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "${root_path}" \
          --data_path "${data_path}" \
          --model_id "${model_prefix}_${seq_len}_${pred_len}_rel" \
          --model "${model_name}" \
          --data "${data_flag}" \
          --features "${features}" \
          --seq_len "${seq_len}" \
          --label_len "${label_len}" \
          --pred_len "${pred_len}" \
          --e_layers "${e_layers}" \
          --d_layers "${d_layers}" \
          --enc_in "${enc_in}" \
          --dec_in "${dec_in}" \
          --c_out "${c_out}" \
          --des 'Exp' \
          --itr 1 \
          "${positional_flag[@]}" \
          "${extra_args[@]}"
      done
    done
  done
}

# ETT datasets
run_transformer_grid "ETTh1_96" "ETTh1" "./dataset/ETT-small/" "ETTh1.csv" "M" 96 48 7 7 7 --factor 3
run_transformer_grid "ETTh2_96" "ETTh2" "./dataset/ETT-small/" "ETTh2.csv" "M" 96 48 7 7 7 --factor 3
run_transformer_grid "ETTm1_96" "ETTm1" "./dataset/ETT-small/" "ETTm1.csv" "M" 96 48 7 7 7
run_transformer_grid "ETTm2_96" "ETTm2" "./dataset/ETT-small/" "ETTm2.csv" "M" 96 48 7 7 7 --factor 1

# Electricity
run_transformer_grid "ECL_96" "custom" "./dataset/electricity/" "electricity.csv" "S" 96 48 1 1 1 --factor 3

# Traffic
run_transformer_grid "traffic_96" "custom" "./dataset/traffic/" "traffic.csv" "M" 96 48 862 862 862 --factor 3 --train_epochs 3

# Weather
run_transformer_grid "weather_96" "custom" "./dataset/weather/" "weather.csv" "M" 96 48 21 21 21 --factor 3 --train_epochs 3

# Exchange
run_transformer_grid "Exchange_96" "custom" "./dataset/exchange_rate/" "exchange_rate.csv" "M" 96 48 8 8 8 --factor 3

# ILI (Illness)
run_transformer_grid "ili_36" "custom" "./dataset/illness/" "national_illness.csv" "M" 36 18 7 7 7 --factor 3
