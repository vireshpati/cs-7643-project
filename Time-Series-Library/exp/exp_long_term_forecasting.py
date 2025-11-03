from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

import wandb

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95),weight_decay=1e-1) #nanoGPT configuration: https://github.com/karpathy/nanoGPT/
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_lr_scheduler(self, optimizer, train_steps):

        warmup_epochs=getattr(self.args, "warmup_epochs", 2)

        def lr_lambda(current_epoch):
            total_steps=self.args.train_epochs * train_steps
            warmup_steps=warmup_epochs * train_steps
            
            #Linear warmup
            if current_epoch < warmup_steps:
                return float(current_epoch) / float(max(1, warmup_steps))
         
            # Linear decay
            progrss = float(current_epoch - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, (1.0-progrss))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
      
        

 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                # Unpack batch - now includes timestamps
                batch_x, batch_y, batch_x_mark, batch_y_mark, timestamps_x, timestamps_y, len_x, len_y = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare timestamps for time-based encodings
                use_timestamps = hasattr(self.args, 'pos_encoding_type') and 'time' in self.args.pos_encoding_type
                timestamps_enc = timestamps_x.float().to(self.device) if use_timestamps else None
                timestamps_dec = timestamps_y.float().to(self.device) if use_timestamps else None

                # decoder input
                # Handle irregular sampling: use actual sequence length instead of fixed label_len/pred_len
                actual_seq_len = batch_y.shape[1]
                if actual_seq_len == self.args.label_len + self.args.pred_len:
                    # Regular sampling case
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    # Irregular sampling case: use actual observed sequence
                    # Split observed sequence proportionally
                    split_point = min(self.args.label_len, actual_seq_len)
                    dec_inp = torch.cat([batch_y[:, :split_point, :],
                                       torch.zeros_like(batch_y[:, split_point:, :])], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if use_timestamps and self.args.model == 'ISaPE':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                               timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if use_timestamps and self.args.model == 'ISaPE':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                           timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Logging config
        wandb_project = self.args.model
        run_name = f"{self.args.model}-{self.args.seq_len}-{self.args.pred_len}"
        if self.args.des != 'test':
            run_name = f"{self.args.model}-{self.args.seq_len}-{self.args.pred_len}-{self.args.des[:10]}"
        wandb.init(
            entity='CS7643F25',
            project=wandb_project,
            name=run_name,
            config=self.args,
        )
        wandb.watch(self.model, log="all", log_freq=100)


        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._get_lr_scheduler(model_optim, train_steps)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                # Unpack batch - now includes timestamps
                batch_x, batch_y, batch_x_mark, batch_y_mark, timestamps_x, timestamps_y, len_x, len_y = batch
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare timestamps for time-based encodings
                use_timestamps = hasattr(self.args, 'pos_encoding_type') and 'time' in self.args.pos_encoding_type
                timestamps_enc = timestamps_x.float().to(self.device) if use_timestamps else None
                timestamps_dec = timestamps_y.float().to(self.device) if use_timestamps else None

                # decoder input
                # Handle irregular sampling: use actual sequence length instead of fixed label_len/pred_len
                actual_seq_len = batch_y.shape[1]
                if actual_seq_len == self.args.label_len + self.args.pred_len:
                    # Regular sampling case
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    # Irregular sampling case: use actual observed sequence
                    # Split observed sequence proportionally
                    split_point = min(self.args.label_len, actual_seq_len)
                    dec_inp = torch.cat([batch_y[:, :split_point, :],
                                       torch.zeros_like(batch_y[:, split_point:, :])], dim=1).float().to(self.device)

                # Random dropping
                drop_mask = None
                if torch.rand(1).item() > 0:
                    random_drop_rate = torch.rand(1).item()
                    drop_mask = torch.rand(1, 1, batch_x.shape[2], device=batch_x.device) < 1-random_drop_rate
                    batch_x = batch_x.masked_fill(drop_mask, 0)
                    batch_y = batch_y.masked_fill(drop_mask, 0)
                    batch_x_mark = batch_x_mark.masked_fill(torch.rand(1, 1, batch_x_mark.shape[2], device=batch_x_mark.device) < 1-random_drop_rate, 0)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if use_timestamps and self.args.model == 'ISaPE':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                               timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if use_timestamps and self.args.model == 'ISaPE':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                           timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            current_lr = model_optim.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "vali_loss": vali_loss,
                "test_loss": test_loss,
                "learning_rate": current_lr,
            })

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                wandb.run.summary["early_stop_epoch"] = epoch + 1
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)  # Disabled: using warmup+linear decay scheduler instead

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        try:
            artifact = wandb.Artifact(run_name, type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Error logging artifact: {e}")






        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
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
            for i, batch in enumerate(test_loader):
                # Unpack batch - now includes timestamps
                batch_x, batch_y, batch_x_mark, batch_y_mark, timestamps_x, timestamps_y, len_x, len_y = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare timestamps for time-based encodings
                use_timestamps = hasattr(self.args, 'pos_encoding_type') and 'time' in self.args.pos_encoding_type
                timestamps_enc = timestamps_x.float().to(self.device) if use_timestamps else None
                timestamps_dec = timestamps_y.float().to(self.device) if use_timestamps else None

                # decoder input
                # Handle irregular sampling: use actual sequence length instead of fixed label_len/pred_len
                actual_seq_len = batch_y.shape[1]
                if actual_seq_len == self.args.label_len + self.args.pred_len:
                    # Regular sampling case
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    # Irregular sampling case: use actual observed sequence
                    # Split observed sequence proportionally
                    split_point = min(self.args.label_len, actual_seq_len)
                    dec_inp = torch.cat([batch_y[:, :split_point, :],
                                       torch.zeros_like(batch_y[:, split_point:, :])], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if use_timestamps and self.args.model == 'ISaPE':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                               timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if use_timestamps and self.args.model == 'ISaPE':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                           timestamps_enc=timestamps_enc, timestamps_dec=timestamps_dec)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        if wandb.run is not None:
            wandb.run.summary["mae"] = mae
            wandb.run.summary["mse"] = mse
            wandb.finish()

        return
