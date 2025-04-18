from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, PhysioNetLoader, SpeechCommandsLoader, CharacterTrajectoriesLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'PhysioNet': PhysioNetLoader,
    'SpeechCommands': SpeechCommandsLoader,
    'CharacterTrajectories': CharacterTrajectoriesLoader
}


def physionet_collate_fn(data, max_len=None):
    # Unpack all three values
    features, labels, masks = zip(*data)
    
    # Convert to tensor and stack
    features = torch.stack(features)  # Shape: [batch_size, 75, 72]
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    
    # Transpose features to [batch_size, 75, 72] if needed
    features = features.permute(0, 2, 1)  # Change to [batch_size, seq_len, features]
    
    return features, labels, masks


def speech_commands_collate_fn(data, max_len=None):
    features, labels, masks = zip(*data)
    
    # Convert to tensor and stack
    features = torch.stack(features)  # Shape: [batch_size, features, seq_len]
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    
    # Transpose to match the format: [batch_size, seq_len, features]
    features = features.permute(0, 2, 1)
    
    # print("Final features shape:", features.shape)
    return features, labels, masks


def character_trajectories_collate_fn(data, max_len=None):
    features, labels, masks = zip(*data)
    
    # Convert to tensor and stack
    features = torch.stack(features)  # Shape: [batch_size, features, seq_len]
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    
    # Transpose to match the format: [batch_size, seq_len, features]
    features = features.permute(0, 2, 1)
    
    return features, labels, masks


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag.lower() == 'test':
        shuffle_flag = False
        # drop_last = True # This might effect the result so we set it to False
        drop_last = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True # This would not effect the reported results as it's not for the test set
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print('load anomaly detection data...')
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        # For SpeechCommands, explicitly pass mfcc parameter
        if args.data == 'SpeechCommands':
            print(f"Creating SpeechCommandsLoader with mfcc={args.mfcc}")
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
                mfcc=args.mfcc  # Explicitly pass mfcc
            )
        # For other datasets like PhysioNet, UEA, etc.
        else:
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
            )
            
        print('load classification data...')
        # print(f"shuffle_flag: {shuffle_flag}")
        shuffle_flag = False # TODO: FOR ERF ANALYSIS, FIX THIS LATER
        
        # Use appropriate collate function based on dataset
        if args.data == 'PhysioNet':
            collate_function = physionet_collate_fn
        elif args.data == 'SpeechCommands':
            collate_function = speech_commands_collate_fn
        elif args.data == 'UEA':
            collate_function = collate_fn
        elif args.data == 'CharacterTrajectories':
            collate_function = character_trajectories_collate_fn
        else:
            collate_function = None

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_function(x, max_len=args.seq_len)
        )
        print(f"drop last bug {flag} {drop_last}")
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        print(f"drop_last_bug after: {flag} {drop_last}")
        return data_set, data_loader
