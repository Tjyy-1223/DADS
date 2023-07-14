from collections import deque
import networkx as nx

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

    # 初始化一个队列 用于bfs遍历
    Q = deque()
    Q.append(initial_node)

    # 开始bfs遍历 -> 构建 pre_node_dict
    while True:
        if len(Q) == 0:
            break
        node = Q.popleft()  # 弹出上一层次的节点

        for neighbor_nodes in nx.neighbors(graph, node):
            if neighbor_nodes == "cloud": continue
            # 如果neighbor_nodes已经在队列里面 就不需要进行重复添加
            if neighbor_nodes not in pre_node_dict.keys() and neighbor_nodes not in Q:
                pre_node_dict[neighbor_nodes] = [node]
                Q.append(neighbor_nodes)
            else:
                pre_node_dict[neighbor_nodes].append(node)
    return pre_node_dict


def transfer_graph_into_model(model ,graph,dict_node_layer, reachable, non_reachable):
    """

    :param model
    :param graph:
    :param dict_node_layer
    :param reachable:
    :param non_reachable:
    :return:
    """
    print(reachable)
    print(non_reachable)

    # edge_layers以及cloud_layers表示在边缘设备上进行推理的layer以及在云端设备上推理的layer
    # 将其保存在字典中并于 node-name 对应
    edge_layers, cloud_layers = {},{}

    for node in reachable:  # 在边缘端处理的层以及映射
        if node == "edge" or node == "v0": continue
        print(node)
        print(dict_node_layer[node] - 1)
        edge_layers[node] = model[dict_node_layer[node] - 1]  # dict_node_layer 表示对应原模型第几层


    for node in non_reachable:  # 在云端处理的层以及映射
        if node == "cloud": continue
        cloud_layers[node] = model[dict_node_layer[node] - 1]  # dict_node_layer 表示对应原模型第几层