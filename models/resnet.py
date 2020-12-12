import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.self_attention import Self_Attn
import torch


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": [models.resnet18(pretrained=False), 512],
                            "resnet50": [models.resnet50(pretrained=False), 2048]}

        resnet, out_ch = self._get_basemodel(base_model)
        # num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        num_ftrs = out_ch * 3 * 3
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs // 2)
        self.l2 = nn.Linear(num_ftrs // 2, out_dim)
        self.att = Self_Attn(out_ch, 'relu')

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        obj_main, obj_bg, attention = self.att(h)

        return self.project(obj_main), self.project(obj_bg)

    def project(self, x):
        x1 = torch.flatten(x, start_dim=1)
        x1 = self.l1(x1)
        x1 = F.relu(x1)
        x1 = self.l2(x1)
        return x1
