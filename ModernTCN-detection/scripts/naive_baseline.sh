# MSL
python -u run.py --task_name anomaly_detection --anomaly_ratio 1 --is_training 1 --root_path ./all_datasets/MSL/ --model_id MSL --model Baseline --data MSL --seq_len 100 --enc_in 55 --c_out 55 --batch_size 128 --des Baseline_Exp --baseline_interval 96 --itr 1

# PSM
python -u run.py --task_name anomaly_detection --anomaly_ratio 1 --is_training 1 --root_path ./all_datasets/PSM/ --model_id PSM --model Baseline --data PSM --seq_len 100 --label_len 0 --pred_len 0 --enc_in 25 --c_out 25 --batch_size 128 --des Baseline_Exp --baseline_interval 100 --itr 1

# SMAP
python -u run.py --task_name anomaly_detection --anomaly_ratio 1 --is_training 1 --root_path ./all_datasets/SMAP/ --model_id SMAP --model Baseline --data SMAP --enc_in 25 --c_out 25 --seq_len 100 --batch_size 128 --des Baseline_Exp --baseline_interval 100 --itr 1

# SMD
python -u run.py --task_name anomaly_detection --anomaly_ratio 0.5 --is_training 1 --root_path ./all_datasets/SMD/ --model_id SMD --model Baseline --data SMD --seq_len 100 --label_len 0 --pred_len 0 --enc_in 38 --c_out 38 --batch_size 128 --des Baseline_Exp --baseline_interval 100 --itr 1

# SWaT
python -u run.py --task_name anomaly_detection --anomaly_ratio 0.5 --is_training 1 --root_path ./all_datasets/SWaT/ --model_id SWAT --model Baseline --data SWAT --seq_len 100 --label_len 0 --pred_len 0 --enc_in 51 --c_out 51 --batch_size 128 --des Baseline_Exp --baseline_interval 100 --itr 1