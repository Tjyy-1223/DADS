from collections import deque
import networkx as nx
from dads_framework.dag_model import DagModel

def get_min_cut_set(graph, min_cut_value, reachable, non_reachable):
    """
    根据最小割算法得到的 min_cut_value, reachable, non_reachable 获取对应的最小割集
    可以根据最小割集帮助 DNN 找到对应的划分层
    :param graph: 构建好的有向图
    :param min_cut_value: 最小割的值，用于assert验证，确保划分正确
    :param reachable: 划分后可以到达的顶点
    :param non_reachable: 划分后不可到达的顶点
    :return: min_cut_set 最小割集，partition_edge表示在DNN模型中的划分点（即不包含 edge 和 cloud 相关的边）
    """
    start = 'edge'
    end = 'cloud'

    cut_set = []
    graph_partition_edge = []
    for u, nbrs in ((n, graph[n]) for n in reachable):
        for v in nbrs:
            if v in non_reachable:
                if u != start and v != end:
                    graph_partition_edge.append((u, v))
                cut_set.append((u, v))

    # 通过 cut-set 得到的最小割值
    cut_set_sum = round(sum(graph.edges[u, v]["capacity"] for (u, v) in cut_set), 3)
    min_cut_value = round(min_cut_value,3)  # 通过 dinic 算法得到的最小割值
    assert cut_set_sum == min_cut_value  # 确保二者相等才可以得正确的划分
    return cut_set, graph_partition_edge

def obtain_pre_node_dict(graph):
    """
    获取 graph 有向图的前置节点信息 pre_node_dict
        key: 当前节点的 node name
        value: 当前节点的前置节点信息，是一个列表（因为可能有多个前置节点）
    :param graph: 传入的有向图
    :return: pre_node_dict 保存当前有向图的前置节点信息
    """
    initial_node = "v0"
    pre_node_dict = {}
    suc_node_nums = {}

    # 初始化一个队列 用于bfs遍历
    Q = deque()
    Q.append(initial_node)

    # 开始bfs遍历 -> 构建 pre_node_dict'
    last_out = None
    while True:
        if len(Q) == 0:
            break
        node = Q.popleft()  # 弹出上一层次的节点

        if "cloud" not in graph[node].keys(): suc_node_nums[node] = len(graph[node])
        else: suc_node_nums[node] = len(graph[node])-1  # 后继节点不考虑cloud
        if suc_node_nums[node] == 0:
            last_out = [node]

        for neighbor_nodes in nx.neighbors(graph, node):
            if neighbor_nodes == "cloud": continue
            # 如果neighbor_nodes已经在队列里面 就不需要进行重复添加
            if neighbor_nodes not in pre_node_dict.keys() and neighbor_nodes not in Q:
                pre_node_dict[neighbor_nodes] = [node]
                Q.append(neighbor_nodes)
            else:
                pre_node_dict[neighbor_nodes].append(node)
    return pre_node_dict, suc_node_nums,last_out


def transfer_graph_into_model(model, dict_node_layer, reachable, non_reachable):
    """

    :param model
    :param dict_node_layer
    :param reachable:
    :param non_reachable:
    :return:
    """
    # edge_layers以及cloud_layers表示在边缘设备上进行推理的layer以及在云端设备上推理的layer
    # 将其保存在字典中并于 node-name 对应
    edge_layers, cloud_layers = {},{}

    for node in reachable:  # 在边缘端处理的层以及映射
        if node == "edge" or node == "v0": continue
        edge_layers[node] = model[dict_node_layer[node] - 1]  # dict_node_layer 表示对应原模型第几层


    for node in non_reachable:  # 在云端处理的层以及映射
        if node == "cloud": continue
        cloud_layers[node] = model[dict_node_layer[node] - 1]  # dict_node_layer 表示对应原模型第几层

    return edge_layers,cloud_layers


def obtain_edge_output_nodes(graph_partition_edge):
    """

    :param graph_partition_edge:
    :return:
    """
    edge_output_nodes = []
    for edge in graph_partition_edge:
        edge_output_nodes.append(edge[0])
    return edge_output_nodes


def edge_cloud_model(model, graph, dict_node_layer, min_cut_value, reachable, non_reachable):
    edge_layers, cloud_layers = transfer_graph_into_model(model, dict_node_layer, reachable, non_reachable)
    pre_node_dict, suc_node_nums, last_out = obtain_pre_node_dict(graph)
    _, graph_partition_edge = get_min_cut_set(graph, min_cut_value, reachable, non_reachable)

    edge_output_nodes = obtain_edge_output_nodes(graph_partition_edge)
    if len(edge_output_nodes) == 0:
        edge_output_nodes = last_out
    cloud_output_nodes = last_out

    edge_model = DagModel(edge_layers, edge_output_nodes,
                          pre_node_dict, suc_node_nums, dict_node_layer)
    cloud_model = DagModel(cloud_layers, cloud_output_nodes,
                           pre_node_dict, suc_node_nums, dict_node_layer)
    return edge_model,cloud_model