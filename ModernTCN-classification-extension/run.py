import argparse
import os
import torch
import torch.optim as optim

from exp.exp_classification import Exp_Classification
import random
import numpy as np
from utils.str2bool import str2bool

parser = argparse.ArgumentParser(description='ModernTCN')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='ModernTCN',
                    help='model name, options: [ModernTCN]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')


# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')



#ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')
parser.add_argument('--use_convffn2', type=str2bool, default=True, help='use ConvFFN2 cross-variable component')

parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage')

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

#multi task
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# classfication task
parser.add_argument('--class_dropout', type=float, default=0.05, help='classfication dropout')

# Add these lines after the classification task arguments
parser.add_argument('--mfcc', type=str2bool, default=True, help='use MFCC features')
parser.add_argument('--sr', type=int, default=16000, 
                    help='sample rate for audio (default: 16000)')
parser.add_argument('--n_mfcc', type=int, default=20,
                    help='number of MFCC coefficients (default: 20)')

# Add these new arguments in the parser section
parser.add_argument('--visualize_erf', action='store_true', 
                    help='visualize effective receptive field', default=False)
parser.add_argument('--weights_path', type=str, default=None,
                    help='path to model weights for ERF visualization')
parser.add_argument('--num_erf_samples', type=int, default=1,
                    help='number of samples to use for ERF visualization')
parser.add_argument('--erf_save_path', type=str, default='erf_scores.npy',
                    help='path to save ERF visualization results')

# Add this argument to the parser
parser.add_argument('--erf_block_idx', type=int, default=None,
                    help='0-indexed block to visualize for ERF (default: None, visualizes after all blocks)')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

def visualize_erf(args, model):
    print("Starting ERF visualization...")
    model.eval()
    
    class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.avg = 0
            self.sum = 0
            self.count = 0
        def update(self, val):
            self.sum = self.sum + val if self.count == 0 else np.add(self.sum, val)
            self.count += 1
            self.avg = self.sum / self.count
    
    meter = AverageMeter()
    
    from data_provider.data_factory import data_provider
    dataset, data_loader = data_provider(args, flag='val')
    
    num_samples = min(args.num_erf_samples, len(dataset))
    print(f"Processing {num_samples} samples for ERF visualization")
    
    for i, (x, _, _) in enumerate(data_loader):
        if i >= num_samples:
            break
            
        try:
            # Permute the input data before feeding to the model
            x = x.permute(0, 2, 1)
            print(f"Input shape after permute: {x.shape}")
            
            x = x.cuda() if args.use_gpu else x
            x.requires_grad = True
            
            # Forward pass through feature extractor with the block_idx parameter
            features = model.model.forward_feature(x, block_idx=args.erf_block_idx)
            print(f"Features shape: {features.shape}")
            
            # Get center point in sequence dimension (last dimension)
            center_idx = features.size(3) // 2  # Using size(3) for the sequence dimension (4000)
            
            # Consider each position in the third dimension (32)
            feature_grads = []
            for pos in range(features.size(2)):  # iterates 0 to 31
                # Select center point for this position
                central_feature = features[..., pos, center_idx]
                central_point = torch.nn.functional.relu(central_feature).sum()
                
                # Compute gradient
                grad = torch.autograd.grad(central_point, x, create_graph=False, retain_graph=True)[0]
                grad = torch.nn.functional.relu(grad)
                feature_grads.append(grad)
            
            # Average gradients across all positions
            grad = torch.stack(feature_grads).mean(0)
            
            # Sum across all dimensions except the sequence dimension
            contribution_scores = grad.sum((0, 1)).detach().cpu().numpy()
            print(f"Grad on cpu shape: {grad.detach().cpu().numpy().shape}")
            print(f"Contribution scores shape: {contribution_scores.shape}")
            print(f"Contribution scores: {contribution_scores}")
            
            # Normalize scores
            contribution_scores = contribution_scores / (np.max(np.abs(contribution_scores)) + 1e-6)
            
            if not np.isnan(np.sum(contribution_scores)):
                meter.update(contribution_scores)
                print(f"Processed sample {i+1}/{num_samples}")
            else:
                print(f"Skipping sample {i+1} due to NaN values")
                
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue
    
    # Save the averaged ERF scores
    np.save(args.erf_save_path, meter.avg)
    print(f"ERF scores saved to {args.erf_save_path}")

if __name__ == '__main__':
    if args.task_name == 'classification':
        Exp = Exp_Classification
    if args.large_size[0] < 13:
        args.small_kernel_merged = True

    if args.visualize_erf:
        # Initialize model and visualize ERF
        exp = Exp(args)
        visualize_erf(args, exp.model)
    elif args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_dim{}_nb{}_lk{}_sk{}_ffr{}_ps{}_str{}_merged{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.dims[0],
                args.num_blocks[0],
                args.large_size[0],
                args.small_size[0],
                args.ffn_ratio,
                args.patch_size,
                args.patch_stride,
                args.use_multi_scale,
                args.small_kernel_merged,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
