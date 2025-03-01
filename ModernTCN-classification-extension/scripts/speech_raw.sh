experiment_name=Final_RS_ks_2        
use_mfcc=False
seq_len=16000
enc_in=1
dims=32

# KS 13, 2 blocks
python -u run.py --model_id Raw_Speech_Final_L13 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 13 13 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 13, 3 blocks
python -u run.py --model_id Raw_Speech_Final_L13 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 --large_size 13 13 13 --small_size 5 5 5 --dims $dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 31, 2 blocks
python -u run.py --model_id Raw_Speech_Final_L31 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 31 29 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 31, 3 blocks
python -u run.py --model_id Raw_Speech_Final_L31 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 --large_size 31 29 27 --small_size 5 5 5 --dims $dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 51, 2 blocks
python -u run.py --model_id Raw_Speech_Final_L51 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 51 49 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 51, 3 blocks
python -u run.py --model_id Raw_Speech_Final_L51 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 --large_size 51 49 47 --small_size 5 5 5 --dims $dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 71, 2 blocks
python -u run.py --model_id Raw_Speech_Final_L71 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 --large_size 71 69 --small_size 5 5 --dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

# KS 71, 3 blocks
python -u run.py --model_id Raw_Speech_Final_L71 --des $experiment_name --task_name classification --is_training 1 --root_path ./all_datasets/speech --data SpeechCommands --model ModernTCN --seq_len $seq_len --enc_in $enc_in --ffn_ratio 1 --patch_size 1 --patch_stride 1 --num_blocks 1 1 1 --large_size 71 69 67 --small_size 5 5 5 --dims $dims $dims $dims --head_dropout 0.0 --dropout 0.1 --class_dropout 0.1 --learning_rate 0.001 --batch_size 32 --train_epochs 100 --patience 10 --use_multi_scale False --mfcc $use_mfcc --itr 5

