
from models.AlexNet import AlexNet
from models.VggNet import VggNet
from models.InceptionBlock import InceptionBlock
from models.IncetiopnBlockV2 import InceptionBlockV2
from models.EasyModel import EasyModel

import torch

if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    print(str(x[0][0][0][:5]))

