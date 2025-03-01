# COMMANDS TO CREATE ERF SCORES FOR MFCC DATA, THEN RUN analyze_erf.py TO VISUALIZE

# KS 13, 3 blocks MFCC, for block 0. You can change the block index to visualize other blocks. It is 0-indexed.
python -u run.py \
    --model_id Speech_Final_L13 \
    --des $experiment_name \
    --task_name classification \
    --is_training 0 \
    --root_path ./all_datasets/speech \
    --data SpeechCommands \
    --model ModernTCN \
    --seq_len $seq_len \
    --enc_in $enc_in \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 1 \
    --large_size 13 13 13 \
    --small_size 5 5 5 \
    --dims $dims $dims $dims \
    --head_dropout 0.0 \
    --dropout 0.0 \
    --class_dropout 0.0 \
    --use_multi_scale False \
    --mfcc True \
    --visualize_erf \
    --erf_block_idx 0 \
    --weights_path checkpoints/Speech_Final_L13_ModernTCN_SpeechCommands_ftM_sl161_pl0_dim32_nb1_lk13_sk5_ffr1_ps1_str1_multiFalse_mergedFalse_Final_S_ks_3_2 \
    --erf_save_path erf_scores.npy --batch_size 1 --num_erf_samples 50

# KS 31, 3 blocks MFCC, block 0. You can change the block index to visualize other blocks. It is 0-indexed.
python -u run.py \
    --model_id Speech_Final_L31 \
    --des $experiment_name \
    --task_name classification \
    --is_training 0 \
    --root_path ./all_datasets/speech \
    --data SpeechCommands \
    --model ModernTCN \
    --seq_len $seq_len \
    --enc_in $enc_in \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 1 \
    --large_size 31 29 27 \
    --small_size 5 5 5 \
    --dims $dims $dims $dims \
    --head_dropout 0.0 \
    --dropout 0.0 \
    --class_dropout 0.0 \
    --use_multi_scale False \
    --mfcc True \
    --visualize_erf \
    --erf_block_idx 0 \
    --weights_path checkpoints/Speech_Final_L31_ModernTCN_SpeechCommands_ftM_sl161_pl0_dim32_nb1_lk31_sk5_ffr1_ps1_str1_multiFalse_mergedFalse_Final_S_ks_3_2 \
    --erf_save_path erf_scores.npy --batch_size 1 --num_erf_samples 50

# KS 51, 3 blocks MFCC, block 0. You can change the block index to visualize other blocks. It is 0-indexed.
python -u run.py \
    --model_id Speech_Final_L51 \
    --des $experiment_name \
    --task_name classification \
    --is_training 0 \
    --root_path ./all_datasets/speech \
    --data SpeechCommands \
    --model ModernTCN \
    --seq_len $seq_len \
    --enc_in $enc_in \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 1 \
    --large_size 51 49 47 \
    --small_size 5 5 5 \
    --dims $dims $dims $dims \
    --head_dropout 0.0 \
    --dropout 0.0 \
    --class_dropout 0.0 \
    --use_multi_scale False \
    --mfcc True \
    --visualize_erf \
    --erf_block_idx 0 \
    --weights_path checkpoints/Speech_Final_L51_ModernTCN_SpeechCommands_ftM_sl161_pl0_dim32_nb1_lk51_sk5_ffr1_ps1_str1_multiFalse_mergedFalse_Final_S_ks_3_2 \
    --erf_save_path erf_scores.npy --batch_size 1 --num_erf_samples 50

# KS 71, 3 blocks MFCC
python -u run.py \
    --model_id Speech_Final_L71 \
    --des $experiment_name \
    --task_name classification \
    --is_training 0 \
    --root_path ./all_datasets/speech \
    --data SpeechCommands \
    --model ModernTCN \
    --seq_len $seq_len \
    --enc_in $enc_in \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 1 \
    --large_size 71 69 67 \
    --small_size 5 5 5 \
    --dims $dims $dims $dims \
    --head_dropout 0.0 \
    --dropout 0.0 \
    --class_dropout 0.0 \
    --use_multi_scale False \
    --mfcc True \
    --visualize_erf \
    --erf_block_idx 0 \
    --weights_path checkpoints/Speech_Final_L71_ModernTCN_SpeechCommands_ftM_sl161_pl0_dim32_nb1_lk71_sk5_ffr1_ps1_str1_multiFalse_mergedFalse_Final_S_ks_3_2 \
    --erf_save_path erf_scores.npy --batch_size 1 --num_erf_samples 50