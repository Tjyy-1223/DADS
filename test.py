


import torch
import networkx as nx
import matplotlib.pyplot as plt
from dads_framework.dag_model import DagModel

from utils import inference_utils
from dads_framework.dinic  import dinic_algorithm
import dads_framework.dads as dads
from dads_framework.graph_construct import graph_construct
from dads_framework.inference import edge_cloud_model
if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    # # print(str(x[0][0][0][:5]))
    # print(len(x.shape))
    # print(x.shape)
    #
    # y = torch.rand(x.shape)d
    # print(y.shape)

    model = inference_utils.get_dnn_model("inception")
    print(len(model))
    graph, dict_node_layer, dict_layer_input_size = graph_construct(model, x, bandwidth=10)
    min_cut_value, reachable, non_reachable = dinic_algorithm(graph)
    model = model.to("cuda")

    edge_model, cloud_model = edge_cloud_model(model,graph,dict_node_layer,min_cut_value, reachable, non_reachable)
    edge_model = edge_model.to("cuda")
    # print(next(model.parameters()).is_cuda)

    edge_model = edge_model.cuda()
    print(next(edge_model.parameters()).device)
    # edge_model.info()
    mid_x = edge_model(x)
    for x in mid_x.keys():
        print(x,mid_x[x].shape)
    # print(mid_x)
    # print(mid_x)
    print("========================")
    # cloud_model.info()
    end_x = cloud_model(mid_x)

    for x in end_x.keys():
        print(x,end_x[x].shape)



