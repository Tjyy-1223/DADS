import torch
import torch.nn as nn
from graph_construct import graph_construct
from dinic import dinic_algorithm,get_min_cut_set


def algorithm_DSL(model, model_input, bandwidth, net_type="wifi"):
    """
    在低负载情况下为传入模型选择最优分割策略
    :param model: 传入DNN模型
    :param model_input: 模型输入
    :param bandwidth: 网络带宽 MB/s
    :param net_type: 当前网络带宽状况，默认为 "wifi"
    :return: 有向图中的对应的割集（不包含edge顶点和cloud顶点）以及划分过程会用到的 dict_node_layer，记录了顶点对应了第几层
    """
    # 构建对应的有向图
    graph, dict_node_layer, dict_layer_input_size = graph_construct(model, model_input, bandwidth=bandwidth, net_type=net_type)
    # min_cut_value表示最短推理时延，reachable表示需要放在边缘端推理的顶点， non_reachable表示放在云端推理的顶点
    min_cut_value, reachable, non_reachable = dinic_algorithm(graph)
    # partition_edge表示图中需要切割的边
    _, graph_partition_edge = get_min_cut_set(graph, min_cut_value, reachable, non_reachable)
    return graph_partition_edge,dict_node_layer



def get_partition_points(graph_partition_edge, dict_node_layer):
    """
    根据有向图的割集 graph_partition_edge 转换成DNN模型切分点 model_partition_edge
    :param graph_partition_edge: 有向图的割集
    :param dict_node_layer: 有向图顶点与模型层的对应
    :return: model_partition_edge:模型中在哪两层之间进行分割
    """
    model_partition_edge = []
    for graph_edge in graph_partition_edge:
        # 表示在DNN模型中的第 start_layer 层 - end_layer之间进行划分(也就是说在start_layer之后进行划分)
        start_layer = dict_node_layer[graph_edge[0]]
        end_layer = dict_node_layer[graph_edge[1]]
        model_partition_edge.append((start_layer, end_layer))
    return model_partition_edge



def model_partition(model, model_partition_edge):
    """
    根据 model_partition_edge 对DNN模型-model进行划分
    :param model: 传入的DNN模型
    :param model_partition_edge:模型分层点
    :return: 边缘端模型 edge_model, 云端模型 cloud_model
    """
    # 如果 model_partition_edge 是[]，代表模型全部部署在边缘执行
    if len(model_partition_edge) == 0:
        return model,nn.Sequential()

    model_partition_points = []  # 记录对应模型的划分点
    for layer_edge in model_partition_edge:
        # 取出在第几层之后对模型进行划分（在第start_layer_index层之后进行划分）
        start_layer_index, end_layer_index = layer_edge[0], layer_edge[1]
        # 记录从模型第几层之后进行划分
        model_partition_points.append(start_layer_index)

    # 开始构建边端模型和云端模型
    edge_model, cloud_model = nn.Sequential(), nn.Sequential()
    if len(model_partition_points) == 1:  # 表示划分与DAG拓扑没有关系 使用链式结构的划分
        idx = 1
        for layer in model:
            if idx <= model_partition_points[0]:  edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            else:  cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            idx += 1
        return edge_model, cloud_model
    else:
        return None,None


def algorithm_dads(model, model_input, bandwidth, net_type="wifi"):
    """
    为DNN模型选择最优划分策略，即将 图中顶点 partition_edge --转化为--> DNN模型第几层进行划分
    :param model: 传入DNN模型
    :param model_input: 模型输入
    :param bandwidth: 网络带宽 MB/s
    :param net_type: 当前网络带宽状况，默认为 "wifi"
    :return: 划分后的边缘模型以及云端模型
    """
    # 获得图中的割集以及dict_node_layer字典
    graph_partition_edge, dict_node_layer = algorithm_DSL(model, model_input, bandwidth, net_type)
    # 获得在DNN模型哪层之后划分
    model_partition_edge = get_partition_points(graph_partition_edge, dict_node_layer)