from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, calculate_metrics
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
from models.ModernTCN import ReparamLargeKernelConv
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        
        # print(f"Feature DataFrame shape: {train_data.feature_df.shape}")
        # print(f"Setting enc_in to: {train_data.feature_df.shape[1]}")
        
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()

        # Calculate and print the number of parameters PER LAYER
        print("Parameter count per layer:")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, ReparamLargeKernelConv)): # Add other layer types if needed
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"\tLayer: {name}, Type: {type(module).__name__}, Parameters: {num_params}")

        # Calculate and print the total number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal number of trainable parameters: {total_params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        # print(f"Data shape in PhysioNetLoader: {data_set.data.shape}")  # Should be [n_samples, 75, 72]
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        metrics = calculate_metrics(predictions, probs.cpu().numpy(), trues)

        self.model.train()
        return total_loss, metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, args=self.args)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            if self.args.data == 'PhysioNet':
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali AUC: {5:.3f} Vali Prec: {6:.3f} Test Loss: {7:.3f} Test Acc: {8:.3f} Test AUC: {9:.3f} Test Prec: {10:.3f}"
                    .format(epoch + 1, train_steps, train_loss, vali_loss, val_metrics['accuracy'], val_metrics['roc_auc'], val_metrics['precision_macro'],
                            test_loss, test_metrics['accuracy'], test_metrics['roc_auc'], test_metrics['precision_macro']))
            else:
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali Prec: {5:.3f} Test Loss: {6:.3f} Test Acc: {7:.3f} Test Prec: {8:.3f}"
                    .format(epoch + 1, train_steps, train_loss, vali_loss, val_metrics['accuracy'], val_metrics['precision_macro'],
                            test_loss, test_metrics['accuracy'], test_metrics['precision_macro']))
            
            # Pass the appropriate metric based on the task
            metric_to_track = val_metrics['roc_auc'] if self.args.data == 'PhysioNet' else val_metrics['accuracy']
            early_stopping(-metric_to_track, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim,  scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        metrics = calculate_metrics(predictions, probs.cpu().numpy(), trues)

        print('Accuracy: {:.3f}'.format(metrics['accuracy']))
        print('ROC AUC: {:.3f}'.format(metrics['roc_auc']))
        print('Precision (macro): {:.3f}'.format(metrics['precision_macro']))

        experiment_name = self.args.des
        folder_path = './results_txt/'
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        f = open(f"results_txt/result_classification_{experiment_name}.txt", 'a')
        f.write(setting + "  \n")
        f.write('Accuracy: {:.3f}\n'.format(metrics['accuracy']))
        f.write('ROC AUC: {:.3f}\n'.format(metrics['roc_auc']))
        f.write('Precision (macro): {:.3f}\n'.format(metrics['precision_macro']))
        f.write('\n')
        f.close()
        return
