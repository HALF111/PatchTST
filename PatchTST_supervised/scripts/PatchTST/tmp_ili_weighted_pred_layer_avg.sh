if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
# seq_len=36

# model_name=PatchTST
# model_name=PatchTST_MoE
# model_name=PatchTST_multi_MoE
# model_name=PatchTST_head_MoE
model_name=PatchTST_weighted_pred_layer_avg

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

gpu_num=3

random_seed=2021
# for pred_len in 24 36 48 60
# for seq_len in 60 80 104 144
# for seq_len in 60
# for seq_len in 60 120 168 240
# for seq_len in 120 168 192
# for seq_len in 240 280 336
for seq_len in 168
do
for pred_len in 24
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'constant'\
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.0025 \
      --d_pred 512 \
      --run_test \
      --gpu $gpu_num \
    #   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done