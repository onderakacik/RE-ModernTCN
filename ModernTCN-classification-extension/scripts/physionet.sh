# MODERNTCN, OPTIMAL:
# 4 layers, lradj type 3, dims 32, kernel 31
experiment_name=PN_MTCN_Tuned
seq_len=72
enc_in=75
dims=32

python -u run.py --model_id PN_MTCN --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 1 --large_size 31 29 27 25 --small_size 5 5 5 5 --dims $dims $dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --lradj type3 --itr 5


# MODERNTCN, ALL VAR. INDEPENDENT:
# 4 layers, lradj type 3, dims 32, kernel 31, use_convffn2 False
experiment_name=PN_MTCN_Tuned
seq_len=72
enc_in=75
dims=32
python -u run.py --model_id PN_MTCN_var_indep --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 1 --large_size 31 29 27 25 --small_size 5 5 5 5 --dims $dims $dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --lradj type3 --use_convffn2 False --itr 5


# MODERNTCN, EMBEDDING VAR. DEPENDENT (MIXED VARIABLES):
# 4 layers, lradj type 3, dims 32, kernel 31
experiment_name=PN_MTCN_Tuned
seq_len=72
enc_in=75
dims=32

python -u run.py --model_id PN_MTCN_embedding_var_dep --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 1 --large_size 31 29 27 25 --small_size 5 5 5 5 --dims $dims $dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --lradj type3 --mix True --itr 5


# CKCONV+MODERNTCN, EMBEDDING VAR. DEPENDENT (MIXED VARIABLES):
# 4 layers, lradj type 3, dims 32, kernel 31
experiment_name=PN_MTCN_Tuned
seq_len=72
enc_in=75
dims=16

python -u run.py --model_id PN_CKConv_MTCN_embedding_var_dep --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 1 --large_size 31 29 27 25 --small_size 5 5 5 5 --dims $dims $dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --lradj type3  --mix True --conv_type CKConv --causal True --separable False --itr 5


# CKCONV+MODERNTCN, EMBEDDING VAR. INDEPENDENT (NO MIXING):
# 4 layer, lradj type 3, dims 32, kernel 13
experiment_name=PN_MTCN_Tuned
seq_len=72
enc_in=75
dims=32

python -u run.py --model_id PN_CKConv_MTCN_optimal --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/sepsis --data PhysioNet --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 1 --large_size 13 13 13 13 --small_size 5 5 5 5 --dims $dims $dims $dims $dims --head_dropout 0.0 --dropout 0.3 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --lradj type3 --itr 5 --mix False --conv_type CKConv --causal True --separable False