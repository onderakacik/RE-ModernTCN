# Prediction Length 6
# First variation: seq_len = 12
python -u run.py --des ETTm1_pl6 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 12 --pred_len 6 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 7 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Second variation: seq_len = 18
python -u run.py --des ETTm1_pl6 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 18 --pred_len 6 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Third variation: seq_len = 24
python -u run.py --des ETTm1_pl6 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 24 --pred_len 6 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Prediction Length 12
# First variation: seq_len = 24
python -u run.py --des ETTm1_pl12 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 24 --pred_len 12 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Second variation: seq_len = 36
python -u run.py --des ETTm1_pl12 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 36 --pred_len 12 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Third variation: seq_len = 48
python -u run.py --des ETTm1_pl12 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 48 --pred_len 12 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Prediction Length 18
# First variation: seq_len = 36
python -u run.py --des ETTm1_pl18 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 36 --pred_len 18 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Second variation: seq_len = 54
python -u run.py --des ETTm1_pl18 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 54 --pred_len 18 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

# Third variation: seq_len = 72
python -u run.py --des ETTm1_pl18 --model_id ETTm1 --is_training 1 --model ModernTCN --root_path ./all_six_datasets/ETT-small --data_path ETTm1.csv --data ETTm1 --features M --seq_len 72 --pred_len 18 --ffn_ratio 8 --patch_size 8 --patch_stride 4 --num_blocks 3 --large_size 13 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.3 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --lradj type3 --use_multi_scale False --small_kernel_merged False --label_len 0 --itr 5

