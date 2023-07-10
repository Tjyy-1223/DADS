import networkx as nx
import sys
import pickle

sys.path.append("../DNN_Architecture")
sys.path.append("../latency_predictor")
sys.path.append("../Net_Transmission")
sys.path.append("../work2_Edgent")
from collections.abc import Iterable
import time
import MobileNet
import MobileNet2
import ResNet
import ResNet2
import GoogLeNet
import GoogLeNet2
import Inceptionv2

import predictor  # 用于预测计算时延
import speed_validate

inf = sys.maxsize
construction_time = 0.0


def graph_construct(model, input, bandwidth, net_type="wifi"):
    """
    传入一个DNN模型，construct_digraph_by_model将DNN模型构建成具有相应权重的有向图
    构建过程主要包括三个方面：
    (1) 从边缘设备-dnn层的边 权重设置为云端推理时延
    (2) dnn层之间的边 权重设置为传输时延
    (3) 从dnn层-云端设备的边 权重设置为边端推理时延
    :param model: 传入dnn模型
    :param input: dnn模型的初始输入
    :param bandwidth: 当前网络时延带宽，可由带宽监视器获取
    :return: 构建好的有向图graph, dict_vertex_layer, dict_layer_input

    由于 GoogleNet 和 ResNet 不能用简单地 x = layer(x) 进行下一步执行
    所以需要自定义新的 get_min_cut_value_for_ResBlock
    所以用户如果有新的DAG结构 （1）完善已有创建结构 （2）iterable api 需要自定义
    """
    graph = nx.DiGraph()

    """
    dict_for_input 字典的作用：
        :key tuple (input.size,input_slice) 字典的键是 输入的形状以及输入的切片(取输入中的前3个数据)
        :value 与之对应的构建好的有向图中的顶点 node_name
    通过dict_for_input可以将 DNN layer 转化为有向图中的顶点 node_name
    原理：对于每一个DNN中的layer 其输入数据是唯一的
    """
    dict_input_size_node_name = {}

    """
    dict_vertex_layer 字典的作用：
        :key node_name 有向图中顶点的名称
        :value 对应原DNN中第几层 layer_index
    可以通过有向图的顶点 node_name 找到其对应原DNN模型中第几层
    注意：
        layer_index = 0 代表初始输入 
        layer_index > 0 表示目前顶点代表原DNN层的第layer_index层，若想取出原DNN层应使用 model[layer_index-1]
    """
    dict_node_layer = {"v0": 0}  # 初始化v0对应的为初始输入

    """
    dict_layer_input_size 字典的作用：
        :key 原DNN中第几层 layer_index
        :value DNN中第 layer_index 的层输入大小
    可以用于查找原模型中第 layer_index 层的输入是什么
    注意：
        layer_index = 0 代表初始输入 
        layer_index = n 获取的是原模型中 model[layer_index-1] 层的输入
    """
    dict_layer_input = {0: None}  # 初始化第0层的大小为初始输入input.shape

    cloud_vertex = "cloud"  # 云端设备节点
    edge_vertex = "edge"  # 边缘设备节点

    print(f"start construct graph for model...")
    graph.add_edge(edge_vertex, "v0", capacity=inf)  # 构建模型初始输入v0
    vertex_index = 0  # 构建图的顶点序号

    for layer_index, layer in enumerate(model):
        # 获取当前层在边缘端设备上的推理时延以及在云端设备上的推理时延
        edge_lat = 5
        cloud_lat = 5
        # edge_lat = predictor.predict_layer_latency(layer,x,edge_device=True)
        # cloud_lat = predictor.predict_layer_latency(layer,x,edge_device=False)

        # 获取当前层需要的传输时延
        # transmission_lat = get_transmission_lat(x,network_type=net_type,define_speed=define_speed)
        transmission_lat = 5

        # 一层dnn layer可以构建一条边，而构建一条边需要两个顶点
        # dict_input_size_node_name 可以根据输入数据大小构建对应的图顶点
        # 所以可以在执行dnn layer的前后分别构建 start_node以及end_node
        vertex_index, start_node = get_node_name(input, vertex_index, dict_input_size_node_name)
        record_input = input
        input = layer(input)
        vertex_index, end_node = get_node_name(input, vertex_index, dict_input_size_node_name)

        # 避免无效层覆盖原始数据 用这种方式可以过滤掉relu层或dropout层
        if start_node == end_node:
            continue  # 不需要进行构建

        # 注意：end_node可以用来在有向图中表示当前的 dnn-layer
        graph.add_edge(edge_vertex, end_node, capacity=cloud_lat)  # 添加从边缘节点到dnn层的边
        graph.add_edge(end_node, cloud_vertex, capacity=edge_lat)  # 添加从dnn层到云端设备的边
        graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 添加从前一个节点到当前节点的边

        dict_node_layer[end_node] = layer_index + 1  # 记录有向图中的顶点对应的DNN的第几层
        dict_layer_input[layer_index + 1] = record_input.size  # 记录DNN层中第i层对应的输入大小

    # 主要负责处理出度大于1的顶点
    prepare_for_partition(graph, vertex_index, dict_node_layer)
    return graph, dict_node_layer, dict_layer_input


def get_node_name(input, vertex_index, dict_input_size_node_name):
    """
    根据输入input构建对应的顶点名称 node_name
    :param input: 当前层的输入
    :param vertex_index: 顶点编号 即目前应该创建哪个顶点
    :param dict_input_size_node_name: 通过dict_for_input可以将 DNN layer 转化为有向图中的顶点 node_name
    :return: node name，构建DAG边所需要的首位节点name
    """
    input_shape = str(input.shape)  # 获取当前input的大小
    input_slice = str(input[0][0][0][:3])  # 获取input的前3个数据，保证数据的唯一性

    if (input_shape, input_slice) not in dict_input_size_node_name.keys():
        node_name = "v" + str(vertex_index)
        dict_input_size_node_name[(input_shape, input_slice)] = node_name  # 创建一个新的节点并保存
        vertex_index += 1
    else:
        node_name = dict_input_size_node_name[(input_shape, input_slice)]  # 从字典中取出原有节点 保证正确构建有向图
    return vertex_index, node_name


def prepare_for_partition(graph, vertex_index, dict_node_layer):
    """
    对根据DNN模型已经构建好的DAG图进行下一步工作：
    1 - 将有多个出点的顶点 记录为start_vex
    2 - 生成新节点为node_name 从node_name -> start_vex 的边代表传输速度，原来从start vex出发的边改为inf
    3 - 找到需要删除的边 ：指原终点为 start vex 的边，将其改成到新节点node name的边
    4 - 删除cloud和edge到原节点的边
    :param graph : 已经构建好的DAG图
    :param vertex_index : 指定下一个生成的节点编号
    :param dict_node_layer : 记录有向图中的顶点对应的DNN的第几层
    :return:
    """
    map_for_vex = []  # 处理 graph  - 1个顶点指向多个其他顶点的情况
    multiple_out_vex = []  # 保存有多个出点的vex
    for edge in graph.edges.data():
        start_vex = edge[0]
        end_vex = edge[1]
        if start_vex == "edge" or end_vex == "cloud":
            continue
        if start_vex not in map_for_vex:  # 如果当前顶点的前置顶点是第一个出现则进行保存
            map_for_vex.append(start_vex)
        elif start_vex not in multiple_out_vex:  # 如果前置顶点已经出现过 再出现的话说明start_vex出度大于1，将其记录在multiple_out_vex中
            multiple_out_vex.append(start_vex)

    for start_vex in multiple_out_vex:
        # 生成新的节点
        node_name = "v" + str(vertex_index)
        vertex_index += 1
        dict_node_layer[node_name] = dict_node_layer[start_vex]  # 新节点与原节点对应原来的同一层

        # 对旧的节点进行改正
        modify_edges = []  # 记录需要修改的边，即起点为start_vex的节点，将其修改为inf
        for edge in graph.edges.data():
            if edge[0] == "edge" or edge[1] == "cloud":
                continue
            if edge[0] == start_vex:
                modify_edges.append(edge)

        # 增加新edge
        for edge in modify_edges:
            graph.add_edge(edge[0], node_name, capacity=edge[2]["capacity"])  # 新增一条从start_vex到node_name的边
            graph.add_edge(node_name, edge[1], capacity=inf)  # 新增从node_name到edge[1]的边 权重为inf
            graph.remove_edge(edge[0],edge[1])  # 删除原有的边

        # 删除 edge - old node
        if graph.has_edge("edge", start_vex):
            data = graph.get_edge_data("edge", start_vex)["capacity"]
            graph.add_edge("edge", node_name, capacity=data)
            graph.remove_edge("edge", start_vex)
        # 删除 old node - cloud
        if graph.has_edge(start_vex, "cloud"):
            data = graph.get_edge_data(start_vex, "cloud")["capacity"]
            graph.add_edge(node_name, "cloud", capacity=data)
            graph.remove_edge(start_vex, "cloud")

    # 简化edge的数值 保留三位小数足够计算
    for edge in graph.edges.data():
        graph.add_edge(edge[0], edge[1], capacity=round(edge[2]["capacity"], 3))
    return vertex_index


from dinic_algorithm import dinic


def get_partition_point(graph, start, end, dict_vertex_layer):
    """
    根据构建好的模型 获得dnn对应的分割点在哪里 为云边协同构建新的协同推理模型
    :param graph: 构建好的DAG架构图
    :param start: min cut 起点
    :param end: min cut 终点
    :param dict_vertex_layer: 字典 存储顶点名称"v1" 与 dnn模型实际层的位置 layer index 1
    :return: cut_set:min cut分割涉及的边，partition_edge : dnn分割涉及的节点，point_list dnn分割点
    """
    # 获得 min cut partition 策略
    # print("start partition ........")
    cut_value = dinic(graph)
    print(f"dinic algorithm value: {cut_value}")
    cut_value, partition = nx.minimum_cut(graph, start, end)
    print(f"inner algorithm value: {cut_value}")
    reachable, non_reachable = partition

    # 获得 min-cut 分割的DNN edge 以及 最小分割涉及的 dege
    cut_set = []
    partition_edge = []
    for u, nbrs in ((n, graph[n]) for n in reachable):
        for v in nbrs:
            if v in non_reachable:
                if u != start and v != end:
                    partition_edge.append((u, v))
                cut_set.append((u, v))

    cut_layer_list = []
    for edge in partition_edge:
        start_layer = dict_vertex_layer[edge[0]]
        end_layer = dict_vertex_layer[edge[1]]
        cut_layer_list.append((start_layer, end_layer))

    cut_set_sum = round(sum(graph.edges[u, v]["capacity"] for (u, v) in cut_set), 3)
    cut_value = round(cut_value, 3)
    # print(f"function get partition point - compare : {cut_set_sum} , {cut_value} , {cut_set_sum == cut_value}")
    # print(cut_set_sum)
    assert cut_set_sum == cut_value

    return cut_set, partition_edge, cut_layer_list


def show_partition_layer(model, cut_layer_list):
    """
    展示从哪个 layer 对模型进行划分
    :param model: 传入模型 方便展示
    :param cut_layer_list: cut layer list 切割点列表
    :return: show cut layer details
    """
    for cut_layer in cut_layer_list:
        start_point = cut_layer[0]
        end_point = cut_layer[1]

        start_layer = "cloud inference" if start_point == 0 else model[start_point - 1]
        end_layer = "cloud inference" if end_point == 0 else model[end_point - 1]

        if start_layer == end_layer:
            print(f"partition after layer {start_point} : \n{start_layer}")
        else:
            print(f"partition from layer {start_point} to layer {end_point}: \n"
                  f"start layer : {start_layer}\n"
                  f"end layer : {end_layer}")
    print("--------------------------------------------------------")


def get_transmission_lat(x, network_type, define_speed):
    """
    根据输入的x获得传输时延 network_type表示网络条件
    3 - WI-FI , 2 - LTE , 1 - 3G
    :param x: 输入的x
    :param network_type: 网络条件状态
    :return: network transmission latency
    """
    # print(network_type)
    transport_size = len(pickle.dumps(x))
    speed, speed_type = speed_validate.get_speed(network_type, define_speed)
    speed_Bpms = speed_validate.get_speed_Bpms(speed=speed, speed_type=speed_type)
    transmission_lat = transport_size / speed_Bpms
    return transmission_lat
