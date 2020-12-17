from util.device import get_device
from models.baseline_encoder import Encoder, Res_encoder
from models.resnet import ResNetSimCLR, get_ResNet34


def get_model(config):
    device = get_device()
    if config['model']['base_model'] == 'baseline':
        model = Encoder(**config["model"]).to(device)
    elif config['model']['base_model'] == 'res_encoder':
        model = Res_encoder(**config["model"]).to(device)
    elif config['model']['base_model'] == 'resnet34':
        model = get_ResNet34()
    else:
        model = ResNetSimCLR(**config['model']).to(device)
    return model
