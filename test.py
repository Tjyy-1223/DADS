
from models.AlexNet import AlexNet
from models.VggNet import VggNet
from models.InceptionBlock import InceptionBlock
from models.IncetiopnBlockV2 import InceptionBlockV2
from models.EasyModel import EasyModel

if __name__ == '__main__':


    model = EasyModel()
    # block = model[8]

    for layer in model:

        print(layer)
        print("----------------")