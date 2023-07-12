
from models.AlexNet import AlexNet
from models.VggNet import VggNet
from models.InceptionBlock import InceptionBlock
from models.IncetiopnBlockV2 import InceptionBlockV2
from models.EasyModel import EasyModel

import torch
import networkx as nx
import matplotlib.pyplot as plt

from dads import graph_construct
from utils import inference_utils
if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    # # print(str(x[0][0][0][:5]))
    # print(len(x.shape))
    # print(x.shape)
    #
    # y = torch.rand(x.shape)
    # print(y.shape)

    model = inference_utils.get_dnn_model("inception")
    graph,_,_ = graph_construct.graph_construct(model,x,bandwidth=10)
    print("......")

    print(graph.nodes)
    print(len(graph.nodes))
    print(len(graph.edges))
    for edge in graph.edges.data():
        print(edge)



