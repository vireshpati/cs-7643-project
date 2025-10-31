export CUDA_VISIBLE_DEVICES=0

model_name=AbsTimeTransformer

pred_len=(96 192 336 720)
e_layers=(2)
d_layers=(1)

for pred_len in ${pred_len[@]}; do
    for e_layers in ${e_layers[@]}; do
        for d_layers in ${d_layers[@]}; do
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTh1.csv \
                --model_id ETTh1_abs_${pred_len} \
                --model $model_name \
                --data ETTh1 \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len ${pred_len} \
                --e_layers ${e_layers} \
                --d_layers ${d_layers} \
                --factor 3 \
                --enc_in 7 \
                --dec_in 7 \
                --c_out 7 \
                --embed absolute \
                --des 'AbsTime' \
                --itr 1
        done
    done
done
