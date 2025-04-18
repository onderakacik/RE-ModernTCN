experiment_name=Final_CT_ks_2
seq_len=182  # Character Trajectories max sequence length
enc_in=3     # Character Trajectories has 3 features (x, y, z coordinates)
dims=32     

# Test different kernel sizes
python -u run.py --model_id CharTraj_Final_L13 \
    --des $experiment_name \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/CharacterTrajectories \
    --data CharacterTrajectories \
    --model ModernTCN \
    --seq_len $seq_len \
    --enc_in $enc_in \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 \
    --large_size 13 13 \
    --small_size 5 5 \
    --dims $dims $dims \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --use_multi_scale False \
    --itr 5