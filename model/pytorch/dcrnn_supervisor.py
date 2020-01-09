import os
import time
import csv

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss, masked_rmse_loss, masked_mse_loss, mixed_mae_mse_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._dataset = kwargs.get('dataset')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter(self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        if self._dataset != 'pems_d7':
            self._data = utils.load_dataset(**self._data_kwargs)
        else:
            self._data = utils.load_dataset_d7(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        log_dir = os.path.join('runs', kwargs.get('base_dir'), log_dir)
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists(os.path.join(self._log_dir, 'models/')):
            os.makedirs(os.path.join(self._log_dir, 'models/'))

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, os.path.join(self._log_dir, 'models/ep%d.tar' % epoch))
        self._logger.info("Saved model at {}".format(epoch))
        return 'ep%d.tar' % epoch

    def load_model(self):
        self._setup_graph()  # NOTE: creating dynamic graph (at model runtime)
        load_path = os.path.join(self._log_dir, 'models/ep%d.tar' % self._epoch_num)
        assert os.path.exists(load_path), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load(load_path, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        # used for updating the dynamically created graph
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()
            if self._dataset != 'pems_d7':
                val_iterator = self._data['val_loader'].get_iterator()
            else:
                val_iterator = self._data['val_loader']

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            if self._dataset != 'pems_d7':
                val_iterator = self._data['val_loader'].get_iterator()
            else:
                val_iterator = self._data['val_loader']
            losses = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def test(self):
        self._logger.info('Start test...')
        with torch.no_grad():
            with open('test_out.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Id', 'Expected'])

                self.dcrnn_model = self.dcrnn_model.eval()

                if self._dataset != 'pems_d7':
                    test_iterator = self._data['test_loader'].get_iterator()
                else:
                    test_iterator = self._data['test_loader']

                preds = {}
                max_f_idx = 0
                for _, (x, file_idx) in enumerate(test_iterator):  # TODO: only for PeMS-D7
                    x = self._prepare_x(x)

                    output = self.dcrnn_model(x)  # (12, bs, #station)
                    output = self.standard_scaler.inverse_transform(output)
                    output = output.permute(1, 0, 2)
                    for f_idx, pred in zip(file_idx, output):  # pred: (12, #station)
                        f_idx = f_idx.item()
                        if f_idx > max_f_idx:
                            max_f_idx = f_idx
                        time_idx = [2, 5, 8]
                        for t in time_idx:
                            for s in range(pred.shape[1]):
                                id = '{}_{}_{}'.format(f_idx, (t+1)*5, s)
                                expected = pred[t, s].item()
                                if not f_idx in preds:
                                    preds[f_idx] = []
                                preds[f_idx].append([id, expected])

                for i in range(max_f_idx + 1):
                    for record in preds[i]:
                        writer.writerow(record)
        self._logger.info('Test done')

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, momentum=0.9, weight_decay=0, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer_type = self._train_kwargs.get('optimizer', 'adam')
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.dcrnn_model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch  # TODO
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):

            self.dcrnn_model = self.dcrnn_model.train()

            if self._dataset != 'pems_d7':
                train_iterator = self._data['train_loader'].get_iterator()
            else:
                train_iterator = self._data['train_loader']
            losses = []

            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x, y, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    if optimizer_type == 'adam':
                        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon, weight_decay=weight_decay)
                    elif optimizer_type == 'sgd':
                        optimizer = torch.optim.SGD(self.dcrnn_model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
                    else:
                        raise NotImplementedError

                loss = self._compute_loss(y, output)

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            lr_scheduler.step()

            self._logger.info("evaluating now!")
            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_loss {:.4f}, val_loss: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            # if (epoch_num % log_every) == log_every - 1:
            #     message = 'Epoch [{}/{}] ({}) train_loss {:.4f}, lr: {:.6f}, ' \
            #               '{:.1f}s'.format(epoch_num, epochs, batches_seen,
            #                                np.mean(losses), lr_scheduler.get_lr()[0],
            #                                (end_time - start_time))
            #     self._logger.info(message)

            # if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
            #     test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
            #     message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, test_loss: {:.4f},  lr: {:.6f}, ' \
            #               '{:.1f}s'.format(epoch_num, epochs, batches_seen,
            #                                np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
            #                                (end_time - start_time))
            #     self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

            if save_model and (epoch_num % 10 == 0 or epoch_num == epochs - 1):
                model_file_name = self.save_model(epoch_num)

            # model_file_name = self.save_model(epoch_num)

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        try:
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
        except:
            pass
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        # x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        # NOTE: truncate input dim
        x = x[..., :self.input_dim].view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _prepare_x(self, x):
        x = self._get_x(x)
        x = self._get_x_in_correct_dims(x)
        return x.to(device)

    def _get_x(self, x):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
        """
        try:
            x = torch.from_numpy(x).float()
        except:
            pass
        self._logger.debug("X: {}".format(x.size()))
        x = x.permute(1, 0, 2, 3)
        return x

    def _get_x_in_correct_dims(self, x):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
        """
        batch_size = x.size(1)
        # x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        # NOTE: truncate input dim
        x = x[..., :self.input_dim].view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        return x

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        loss_type = self._train_kwargs.get('loss', 'mae')
        if loss_type == 'mae':
            return masked_mae_loss(y_predicted, y_true)
        elif loss_type == 'rmse':
            return masked_rmse_loss(y_predicted, y_true)
        elif loss_type == 'mse':
            return masked_mse_loss(y_predicted, y_true)
        elif loss_type == 'mixed':
            # return (masked_mse_loss(y_predicted, y_true) + masked_mae_loss(y_predicted, y_true)) / 2
            return mixed_mae_mse_loss(y_predicted, y_true)
        else:
            raise NotImplementedError

