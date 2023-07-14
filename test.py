
from models.AlexNet import AlexNet
from models.VggNet import VggNet
from models.InceptionBlock import InceptionBlock
from models.IncetiopnBlockV2 import InceptionBlockV2
from models.EasyModel import EasyModel

import torch
import networkx as nx
import matplotlib.pyplot as plt

from utils import inference_utils
from dads_framework.dinic  import dinic_algorithm
import dads_framework.dads as dads
from dads_framework.graph_construct import graph_construct
from dads_framework.collaborative_inference import obtain_pre_node_dict,transfer_graph_into_model
if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    # # print(str(x[0][0][0][:5]))
    # print(len(x.shape))
    # print(x.shape)
    #
    # y = torch.rand(x.shape)
    # print(y.shape)

    model = inference_utils.get_dnn_model("inception")
    print(len(model))
    # graph, dict_node_layer, dict_layer_input_size = graph_construct(model, x, bandwidth=10)
    # min_cut_value, reachable, non_reachable = dinic_algorithm(graph)
    #
    # transfer_graph_into_model(model, graph,dict_node_layer,reachable,non_reachable)

    index = 1
    print(model[10])
    for layer in model:
        print(index,layer)
        index+=1



