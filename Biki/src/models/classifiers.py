import torch
from torch import nn
from torchvision import models

from models.interfaces import TorchModel, Ann, Metrics



class SoudImgClassifier(TorchModel, Metrics):

    def __init__(self, ann_outputsize, ann_input_size=None, ann_num_of_layer=1, ann_step=2, pretrained=True, *args, **kwargs):
        super().__init__()

        self.model = models.resnet18(pretrained=pretrained)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        # Add Ann as last layers
        self.model.fc = Ann(input_size=ann_input_size, output_size=ann_outputsize, step=ann_step, num_of_layer=ann_num_of_layer).model



