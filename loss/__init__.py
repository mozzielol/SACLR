from loss.nt_xent import NTXentLoss
from loss.sim_siam import Siam
from loss.split_loss import Split_KL, Split_KL_view
from util.device import get_device


def get_loss(config):
    if config['loss_func'] == 'sim':
        loss = NTXentLoss(get_device(), config['batch_size'], **config['loss'])
    elif config['loss_func'] == 'siam':
        loss = Siam()
    elif config['loss_func'] == 'split':
        loss = Split_KL(NTXentLoss(get_device(), config['batch_size'], **config['loss']))
    elif config['loss_func'] == 'split_view':
        loss = Split_KL_view(NTXentLoss(get_device(), config['batch_size'], **config['loss']))
    else:
        raise NotImplemented()
    return loss
