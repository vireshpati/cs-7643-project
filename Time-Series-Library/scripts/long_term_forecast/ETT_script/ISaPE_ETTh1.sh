export CUDA_VISIBLE_DEVICES=0

model_name=ISaPE


pos_encoding_list=(abs_index rel_index rope_index abs_time rel_time rope_time) # lot_rope
pred_len_list=(96 192 336 720)

irregular_sampling_list=(none uniform bursty adaptive)


irregular_missing_rate=0.3 # for uniform pattern

irregular_p_miss_to_miss=0.8 # for bursty pattern
irregular_p_obs_to_miss=0.1 # for bursty pattern

irregular_target_retention=0.3  # for adaptive pattern
irregular_window_size=24 # for adaptive pattern


e_layers_list=(2)
n_heads_list=(4)
d_model_list=(64)

dropout_list=(0.3)
batch_size_list=(32)
learning_rate_list=(0.0001)




for pos_encoding_type in "${pos_encoding_list[@]}"; do
  for pred_len in "${pred_len_list[@]}"; do
    for e_layers in "${e_layers_list[@]}"; do
      for n_heads in "${n_heads_list[@]}"; do
        for d_model in "${d_model_list[@]}"; do
          for dropout in "${dropout_list[@]}"; do
            for batch_size in "${batch_size_list[@]}"; do
              for learning_rate in "${learning_rate_list[@]}"; do
                for irregular_sampling_pattern in "${irregular_sampling_list[@]}"; do
                  python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ./dataset/ETT-small/ \
                    --data_path ETTh1.csv \
                    --model_id ETTh1_${pos_encoding_type}_96_${pred_len} \
                    --model $model_name \
                    --data ETTh1 \
                    --features M \
                    --seq_len 96 \
                    --label_len 48 \
                    --pred_len $pred_len \
                    --e_layers $e_layers \
                    --d_model $d_model \
                    --d_ff $((d_model * 4)) \
                    --n_heads $n_heads \
                    --dropout $dropout \
                    --batch_size $batch_size \
                    --learning_rate $learning_rate \
                    --pos_encoding_type $pos_encoding_type \
                    --d_layers 1 \
                    --factor 3 \
                    --enc_in 7 \
                    --dec_in 7 \
                    --c_out 7 \
                    --des 'Exp' \
                    --itr 1 \
                    --train_epochs 100 \
                    --patience 10 \
                    --irregular_sampling_pattern $irregular_sampling_pattern \
                    --irregular_missing_rate $irregular_missing_rate \
                    --irregular_p_miss_to_miss $irregular_p_miss_to_miss \
                    --irregular_p_obs_to_miss $irregular_p_obs_to_miss \
                    --irregular_target_retention $irregular_target_retention \
                    --irregular_window_size $irregular_window_size \
                    --irregular_seed 42 \
                    done
              done
            done
          done
        done
      done
    done
  done
done