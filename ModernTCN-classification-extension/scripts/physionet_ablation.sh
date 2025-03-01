experiment_name=Final_PN_ks_2_ablation
seq_len=72
enc_in=75
dims=128 # DIM = min(max(2 ** math.ceil(math.log2(ENC_IN)), d_min), d_max) => d_min = 32, d_max = 512


python -u run.py --model_id PhysioNet_Final_L13 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 13 13 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --use_convffn2 False --itr 5

python -u run.py --model_id PhysioNet_Final_L31 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 31 29 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --use_convffn2 False --itr 5

python -u run.py --model_id PhysioNet_Final_L51 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 51 49 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --use_convffn2 False --itr 5 

python -u run.py --model_id PhysioNet_Final_L71 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 71 69 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --use_convffn2 False --itr 5
