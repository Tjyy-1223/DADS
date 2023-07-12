
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
import dads.dinic as dinic

if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    # # print(str(x[0][0][0][:5]))
    # print(len(x.shape))
    # print(x.shape)
    #
    # y = torch.rand(x.shape)
    # print(y.shape)

    model = inference_utils.get_dnn_model("easy_net")
    graph, dict_node_layer, dict_layer_input_size = graph_construct.graph_construct(model,x,bandwidth=10)
    min_cut_value, reachable, non_reachable = dinic.dinic_algorithm(graph)

    cut_set, partition_edge = dinic.get_min_cut_set(graph, min_cut_value, reachable, non_reachable)
    print(cut_set)
    print(partition_edge)




# def show_partition_layer(model, cut_layer_list):
#     """
#     展示从哪个 layer 对模型进行划分
#     :param model: 传入模型 方便展示
#     :param cut_layer_list: cut layer list 切割点列表
#     :return: show cut layer details
#     """
#     for cut_layer in cut_layer_list:
#         start_point = cut_layer[0]
#         end_point = cut_layer[1]
#
#         start_layer = "cloud inference" if start_point == 0 else model[start_point - 1]
#         end_layer = "cloud inference" if end_point == 0 else model[end_point - 1]
#
#         if start_layer == end_layer:
#             print(f"partition after layer {start_point} : \n{start_layer}")
#         else:
#             print(f"partition from layer {start_point} to layer {end_point}: \n"
#                   f"start layer : {start_layer}\n"
#                   f"end layer : {end_layer}")
#     print("--------------------------------------------------------")
#
