from util.device import get_device
from models.baseline_encoder import Encoder, Res_encoder
from models.resnet import ResNetSimCLR, get_ResNet34
from models.vgg16 import VGG16


def get_model(config):
    device = get_device()
    base_model = config['model']['base_model']
    if base_model == 'baseline':
        model = Encoder(**config["model"]).to(device)
    elif base_model == 'res_encoder':
        model = Res_encoder(**config["model"]).to(device)
    elif base_model == 'resnet34':
        model = get_ResNet34(config, **config["model"])
    elif base_model == 'vgg16':
        model = VGG16()
    else:
        model = ResNetSimCLR(**config['model']).to(device)
    return model
