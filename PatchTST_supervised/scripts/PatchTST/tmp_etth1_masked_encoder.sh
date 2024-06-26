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
model_name=Masked_encoder

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

gpu_num=1

random_seed=2021
# for seq_len in 96 192 336 720
# for seq_len in 96
# for seq_len in 336
# for seq_len in 336 504 900 1080
# for seq_len in 1200 1360 1600
# for seq_len in 1800 2000 2400
for seq_len in 1200
# for seq_len in 2400
do
# for pred_len in 96 192 336 720
for pred_len in 96
do
    # ! 注意：用masked_encoder时间一定要保证patch之间是non-overlap的！
    # ! 也即stride需要和patch_len一样长！从8变成了16！
    # ! 由于最后不是展平，而是对d_model到patch_len单独映射了，所以感觉d_model是否也要调大一些会更好
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
      --patch_len 16 \
      --stride 16 \
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.015 \
      --gpu $gpu_num \
      --run_train --run_test \
    #   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
done