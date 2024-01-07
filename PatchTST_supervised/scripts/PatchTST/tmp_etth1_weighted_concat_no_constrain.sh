if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336

# model_name=PatchTST
# model_name=PatchTST_multi_MoE
# model_name=PatchTST_head_MoE
# model_name=PatchTST_avg
model_name=PatchTST_weighted_concat_no_constrain

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

gpu_num=1

random_seed=2021
# for seq_len in 96 192 336 720
# for seq_len in 336
for seq_len in 336 504
# for seq_len in 1200 1360 1600
# for seq_len in 1800 2000 2400
# for seq_len in 336
do
# for pred_len in 96 192 336 720
for pred_len in 96
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
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --run_train --run_test \
      --gpu $gpu_num \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done