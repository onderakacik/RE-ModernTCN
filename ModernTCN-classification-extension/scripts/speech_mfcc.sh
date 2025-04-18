experiment_name=Optimal_Speech
use_mfcc=True
seq_len=161
enc_in=20
dims=32
# KS 13, 3 blocks MFCC ==> Optimal
python -u run.py --model_id MFCC_Optimal --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 --large_size 13 13 13 --small_size 5 5 5 --dims $dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --lradj type3 --itr 5

