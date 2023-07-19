from dads_framework.dinic import dinic_algorithm,get_min_cut_set
from dads_framework.graph_construct import graph_construct

def algorithm_DSL(model, model_input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    在低负载情况下为传入模型选择最优分割策略
    :param model: 传入DNN模型
    :param model_input: 模型输入
    :param edge_latency_list: 边缘设备上各层的推理时延
    :param cloud_latency_list: 云端设备上各层的推理时延
    :param bandwidth: 网络带宽 MB/s
    :param net_type: 当前网络带宽状况，默认为 "wifi"
    :return: 有向图中的对应的割集（不包含edge顶点和cloud顶点）以及划分过程会用到的 dict_node_layer，记录了顶点对应了第几层
    """
    # 构建对应的有向图
    graph, dict_node_layer, dict_layer_input_size = graph_construct(model, model_input, edge_latency_list, cloud_latency_list, bandwidth=bandwidth, net_type=net_type)
    # min_cut_value表示最短推理时延，reachable表示需要放在边缘端推理的顶点， non_reachable表示放在云端推理的顶点
    min_cut_value, reachable, non_reachable = dinic_algorithm(graph)

    # 检查一些bug时可能用到
    # for edge in graph.edges(data=True):
    #     print(edge)
    # print(reachable)
    # print(non_reachable)

    # partition_edge表示图中需要切割的边
    graph_partition_edge = get_min_cut_set(graph, min_cut_value, reachable, non_reachable)
    return graph_partition_edge,dict_node_layer



def get_partition_points(graph_partition_edge, dict_node_layer):
    """
    根据有向图的割集 graph_partition_edge 转换成DNN模型切分点 model_partition_edge
    :param graph_partition_edge: 有向图的割集
    :param dict_node_layer: 有向图顶点与模型层的对应
    :return: model_partition_edge: 模型中在哪两层之间进行分割
    """
    model_partition_edge = []
    for graph_edge in graph_partition_edge:
        # 表示在DNN模型中的第 start_layer 层 - end_layer之间进行划分(也就是说在start_layer之后进行划分)
        start_layer = dict_node_layer[graph_edge[0]]
        end_layer = dict_node_layer[graph_edge[1]]
        model_partition_edge.append((start_layer, end_layer))
    return model_partition_edge

