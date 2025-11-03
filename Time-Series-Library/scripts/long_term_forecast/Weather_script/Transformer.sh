export CUDA_VISIBLE_DEVICES=0

model_name=Transformer


pred_len_list=(96 192 336 720)

e_layers_list=(2)
n_heads_list=(8)
d_model_list=(128)

dropout_list=(0.1)
batch_size_list=(64)
learning_rate_list=(0.0001)


for pred_len in "${pred_len_list[@]}"; do
  for e_layers in "${e_layers_list[@]}"; do
    for n_heads in "${n_heads_list[@]}"; do
      for d_model in "${d_model_list[@]}"; do
        for dropout in "${dropout_list[@]}"; do
          for batch_size in "${batch_size_list[@]}"; do
            for learning_rate in "${learning_rate_list[@]}"; do
              python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./dataset/weather/ \
                --data_path weather.csv \
                --model_id weather_96_96 \
                --model $model_name \
                --data custom \
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
                --d_layers 1 \
                --factor 3 \
                --enc_in 21 \
                --dec_in 21 \
                --c_out 21 \
                --des 'Exp' \
                --itr 1 \
                --train_epochs 100 \
                --patience 5 
            done
          done
        done
      done
    done
  done
done
