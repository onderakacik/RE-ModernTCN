import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score, precision_score

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 20 else args.learning_rate * (0.5 ** ((epoch // 20) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: args.learning_rate * 0.5 , 40: args.learning_rate * 0.01, 60:args.learning_rate * 0.01,8:args.learning_rate * 0.01,100:args.learning_rate * 0.01 }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-5, args=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta
        # Set metric name based on data argument
        self.metric_name = 'AUC' if args and args.data == 'PhysioNet' else 'accuracy'
        self.is_auc = args and args.data == 'PhysioNet'

    def __call__(self, val_metric, model, path):
        score = -val_metric  # We negate because we're tracking -metric for minimization
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, path)
        elif score < self.best_score + self.delta:  # Score improved (got smaller)
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # Score got larger (worse)
            self.best_score = score
            self.save_checkpoint(val_metric, model, path)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, path):
        if self.verbose:
            print(f'Validation {self.metric_name} decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_metric_min = val_metric


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def calculate_metrics(predictions, probabilities, labels):
    """
    Calculate classification metrics including ROC AUC and macro-average precision
    Args:
        predictions: numpy array of predicted class indices
        probabilities: numpy array of class probabilities
        labels: numpy array of true labels
    Returns:
        dict: Dictionary containing accuracy, roc_auc, and precision_macro scores
    """
    accuracy = np.mean(predictions == labels)
    
    # Calculate macro-average precision
    precision_macro = precision_score(labels, predictions, average='macro')
    
    # For binary classification, use the probability of class 1
    # For multiclass, calculate ROC AUC in a one-vs-rest fashion
    if probabilities.shape[1] == 2:  # binary case
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
    else:  # multiclass case
        roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr')
        
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision_macro': precision_macro
    }