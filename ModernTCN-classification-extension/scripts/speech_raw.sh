experiment_name=Optimal_Speech
use_mfcc=False
seq_len=16000
enc_in=1
dims=512
# KS 71, 3 blocks ==> Optimal
python -u run.py --model_id Raw_Optimal --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 32 --patch_stride 16 --num_blocks 1 1 1 --large_size 71 69 67 --small_size 5 5 5 --dims $dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --lradj type3 --mfcc $use_mfcc --itr 5
