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
        if (isinstance(model, GoogLeNet.Inception) or isinstance(model, GoogLeNet2.Inception)
            or isinstance(model, Inceptionv2.Inception)
        ) and len(keys) > 1:
            edge_Inception, cloud_Inception = construct_edge_inception(model, block_dict)
            edge_model.add_module(f"{index}-{edge_Inception.__class__.__name__}", edge_Inception)
            index += 1
            cloud_model.add_module(f"{index}-{cloud_Inception.__class__.__name__}", cloud_Inception)
            index += 1
            return index




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




def construct_edge_inception(model,block_dict):
    """
    构建Inception的边端模型和云端模型
    :param model: 传入一个需要划分的Inception block
    :param block_dict: Inception的划分dict
    :return: edge_Inception,cloud_Inception
    """
    assert len(block_dict) == 4
    index = 0
    branches = [model.branch1, model.branch2, model.branch3, model.branch4]
    lens = [len(branches[0]),len(branches[1]),len(branches[2]),len(branches[3])]

    ac_lens = []
    ac_len = 0
    for i in range(4):
        ac_len += lens[i]
        ac_lens.append(ac_len)

    edge_branches = []
    cloud_branches = []
    for edge in block_dict.keys():
        # 选取 edge[0] 为对应划分点
        partition_point = edge[0]
        edge_branch = nn.Sequential()
        cloud_branch = nn.Sequential()

        # 分四种情况对 google branch的四个分支进行处理
        block = None
        if edge[0] in range(0,ac_lens[0] + 1) or edge[1] in range(0,ac_lens[0] + 1):
            block = branches[0]
            if partition_point > 0:  partition_point -= 0
        elif edge[0] in range(ac_lens[0] + 1,ac_lens[1] + 1) or edge[1] in range(ac_lens[0] + 1,ac_lens[1] + 1):
            block = branches[1]
            if partition_point > 0:  partition_point -= ac_lens[0]
        elif edge[0] in range(ac_lens[1] + 1, ac_lens[2] + 1) or edge[1] in range(ac_lens[1] + 1, ac_lens[2] + 1):
            block = branches[2]
            if partition_point > 0:  partition_point -= ac_lens[1]
        elif edge[0] in range(ac_lens[2] + 1, ac_lens[3] + 1) or edge[1] in range(ac_lens[2] + 1, ac_lens[3] + 1):
            block = branches[3]
            if partition_point > 0:  partition_point -= ac_lens[2]


        # 构建对应的 edge branch 以及 cloud branch
        index = construct_edge_model(block, edge_branch, partition_point, index)

        if partition_point < len(block):
            child_dict = block_dict[edge]
            # 取出模型的第 partition_point + 1 层
            child_block = block[partition_point]
            # 递归对child block进行模型分割
            index = get_partition_model(child_block,child_dict,edge_branch,cloud_branch,index)

        index = construct_cloud_model(block,cloud_branch,partition_point+1,index)
        edge_branches.append(edge_branch)
        cloud_branches.append(cloud_branch)

    # 使用 edge_branches 以及 cloud_branches 构建 EdgeInception 以及 CloudInception 两个类
    edge_Inception = GoogLeNet.EdgeInception(edge_branches)
    cloud_Inception = GoogLeNet.CloudInception(cloud_branches)
    return edge_Inception,cloud_Inception
