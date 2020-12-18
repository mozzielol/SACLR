import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.self_attention import Self_Attn, MultiHeadAttention
import torch
from torchvision.models.resnet import ResNet, BasicBlock


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
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        self.l3 = nn.Linear(num_ftrs, num_ftrs)
        self.l4 = nn.Linear(num_ftrs, out_dim)

        self.att = Self_Attn(out_ch, 'relu')
        # self.l3 = nn.Linear(num_ftrs, num_ftrs)
        # self.multi_att = MultiHeadAttention(8, num_ftrs)

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
        # obj_main = torch.flatten(obj_main, start_dim=1)
        # obj_bg = torch.flatten(obj_bg, start_dim=1)
        # obj_main = self.l3(obj_main)
        # obj_bg = self.l3(obj_bg)
        # obj_main = self.multi_att(obj_main, obj_main, obj_main)
        # obj_bg = self.multi_att(obj_bg, obj_bg, obj_bg)

        return self.project_local(obj_main), self.project_global(obj_bg)

    def project_local(self, x):
        x1 = torch.flatten(x, start_dim=1)
        x1 = self.l1(x1)
        x1 = F.relu(x1)
        x1 = self.l2(x1)
        return x1

    def project_global(self, x):
        x1 = torch.flatten(x, start_dim=1)
        x1 = self.l3(x1)
        x1 = F.relu(x1)
        x1 = self.l4(x1)
        return x1


class ResNet34AT(ResNet):
    """Attention maps of ResNet-34.

    Overloaded ResNet model to return attention maps.
    """
    def __init__(self, out_dim, **kwargs):
        super(ResNet34AT, self).__init__(**kwargs)
        num_ftrs = 128 * 12 * 12
        self.multi_att = MultiHeadAttention(8, num_ftrs)
        self.l1 = nn.Linear(num_ftrs, num_ftrs // 2)
        self.l2 = nn.Linear(num_ftrs // 2, out_dim)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g1 = self.layer3(g1)
        # g3 = self.layer4(g2)
        # obj_main = self.multi_att(g0, g0, g0)
        g1 = torch.flatten(g1, start_dim=1)
        obj_bg = self.multi_att(g1, g1, g1)
        return self.project(obj_bg.squeeze()), self.project(g1),  # [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

    def project(self, x):
        x1 = torch.flatten(x, start_dim=1)
        x1 = self.l1(x1)
        x1 = F.relu(x1)
        x1 = self.l2(x1)
        return x1


def get_ResNet34(base_model, out_dim):
    return ResNet34AT(out_dim, **{'block': BasicBlock, 'layers': [3, 4, 6, 3]})