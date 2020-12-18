import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.sim_siam import Siam
import os
import shutil
import sys
from util.device import get_device
from models.baseline_encoder import Encoder
from models.resnet import ResNetSimCLR
from models.model_util import get_model


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SaCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        if config['loss_func'] == 'sim':
            self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        elif config['loss_func'] == 'siam':
            self.siam_loss = Siam(config['batch_size'])
        else:
            raise NotImplemented()


    def _step(self, model, train_x):
        # get the representations and the projections,
        # Currently, get the attention of the representation
        obj_main, obj_bg = model(train_x)  # [N,C]
        # normalize projection feature vectors
        zis = F.normalize(obj_main, dim=1)
        zjs = F.normalize(obj_bg, dim=1)

        if self.config['loss_func'] == 'sim':
            loss = self.nt_xent_criterion(zis, zjs)
        elif self.config['loss_func'] == 'siam':
            loss = self.siam_loss(obj_main, obj_bg)
        else:
            raise ValueError('loss not valid ')

        return loss, obj_main

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = get_model(self.config).to(get_device())
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, self.config['ckpt'])

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print('Training begin ...')
        for epoch_counter in range(self.config['epochs']):
            for train_x, _ in train_loader:
                optimizer.zero_grad()

                train_x = train_x.to(self.device)

                loss, _ = self._step(model, train_x)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as loss:
                        loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1
                print('Epoch {}/{}, training loss: {:.4f}'
                      .format(epoch_counter, self.config['epochs'], loss.item()))
            print('')

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step(epoch_counter)
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_iter=0):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for val_x, cls in valid_loader:
                val_x = val_x.to(self.device)
                loss, obj_main = self._step(model, val_x)
                valid_loss += loss.item()
                if counter == 0:
                    self.writer.add_embedding(torch.flatten(obj_main, start_dim=1),
                                              metadata=cls,
                                              label_img=val_x, global_step=n_iter)
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
