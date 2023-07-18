import networkx as nx
import sys
from predictor.predictor_utils import predict_model_latency
from net.net_utils import get_speed
import pickle

inf = sys.maxsize
construction_time = 0.0
predictor_dict = {}

def add_graph_edge(graph, vertex_index, input, layer_index, layer, bandwidth, net_type,
                 dict_input_size_node_name, dict_node_layer, dict_layer_input_size, dict_layer_output,
                 record_flag):
    """
    向一个有向图中添加
    :param graph: 向哪个有向图中添加
    :param vertex_index: 当前构建的顶点编号
    :param input: 当前层的输入
    :param layer_index: 当前层
    :param layer: 当前层类型
    :param bandwidth: 网络带宽
    :param net_type: 网络类型
    :param dict_input_size_node_name:   字典：key:输入 value:对应的顶点编号
    :param dict_node_layer:             字典：key:顶点编号 value:对应DNN中第几层
    :param dict_layer_input_size:       字典：key:DNN中第几层 value:对应的输入大小
    :param dict_layer_output:            字典：key:DNN中第几层 value:对应的输出
    :param record_flag: 只有某些关键层才会记录层的输出
    :return: 当前构建的顶点数目 vertex_index ，以及当前层的输出（会用于作为下一层的输入）
    """
    cloud_vertex = "cloud"  # 云端设备节点
    edge_vertex = "edge"  # 边缘设备节点

    # 获取当前层在边缘端设备上的推理时延以及在云端设备上的推理时延
    edge_lat = predict_model_latency(input, layer, device="edge", predictor_dict=predictor_dict)
    cloud_lat = predict_model_latency(input, layer, device="cloud", predictor_dict=predictor_dict)

    # 获取当前层需要的传输时延
    #   predict transmission latency,network_type = WI-FI
    transport_size = len(pickle.dumps(input))
    speed = get_speed(network_type=net_type,bandwidth=bandwidth)
    transmission_lat = transport_size / speed

    # 一层dnn layer可以构建一条边，而构建一条边需要两个顶点
    # dict_input_size_node_name 可以根据输入数据大小构建对应的图顶点
    # 所以可以在执行dnn layer的前后分别构建 start_node以及end_node
    start_node, end_node, record_input = None, None, None

    if isinstance(input,list):
        layer_out = None
        record_input = input
        for one_input in input:
            vertex_index, start_node = get_node_name(one_input, vertex_index, dict_input_size_node_name)
            layer_out = layer(input)
            vertex_index, end_node = get_node_name(layer_out, vertex_index, dict_input_size_node_name)

            # 例如 input 是长度为n的列表，则需要构建n个边
            graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 添加从前一个节点到当前节点的边
        input = layer_out
    else:  # 常规构建
        vertex_index, start_node = get_node_name(input, vertex_index, dict_input_size_node_name)
        record_input = input
        input = layer(input)
        vertex_index, end_node = get_node_name(input, vertex_index, dict_input_size_node_name)

        # 避免无效层覆盖原始数据 用这种方式可以过滤掉relu层或dropout层
        if start_node == end_node:
            return vertex_index,input  # 不需要进行构建
        graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 添加从前一个节点到当前节点的边

    # 注意：end_node可以用来在有向图中表示当前的 dnn-layer
    graph.add_edge(edge_vertex, end_node, capacity=cloud_lat)  # 添加从边缘节点到dnn层的边
    graph.add_edge(end_node, cloud_vertex, capacity=edge_lat)  # 添加从dnn层到云端设备的边

    dict_node_layer[end_node] = layer_index + 1  # 记录有向图中的顶点对应的DNN的第几层
    # dict_layer_input_size[layer_index + 1] = record_input.shape  # 记录DNN层中第i层对应的输入大小
    if record_flag:
        dict_layer_output[layer_index+1] = input  # 记录DNN层中第i层对应的输出

    return vertex_index,input



def graph_construct(model, input, bandwidth, net_type="wifi"):
    """
    传入一个DNN模型，construct_digraph_by_model将DNN模型构建成具有相应权重的有向图
    构建过程主要包括三个方面：
    (1) 从边缘设备-dnn层的边 权重设置为云端推理时延
    (2) dnn层之间的边 权重设置为传输时延
    (3) 从dnn层-云端设备的边 权重设置为边端推理时延
    :param model: 传入dnn模型
    :param input: dnn模型的初始输入
    :param bandwidth: 当前网络时延带宽，可由带宽监视器获取 MB/s
    :param net_type: 当前网络类型 默认为 wifi
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
    dict_layer_input 以及 dict_layer_output 字典的作用：
        :key 原DNN中第几层 layer_index 
        :value DNN中第 layer_index 的层输入以及输出是什么
    第 layer_index 层的输入与输出，可以使用 shape 以及前三个元素确定是否为同1输入
    注意：
        layer_index = 0 代表初始输入 
        layer_index = n 获取的是原模型中 model[layer_index-1] 层的输入
    """
    dict_layer_input = {0: None}  # 第0层为初始输入 其输入记录为None
    dict_layer_output = {0: input}  # 第0层为初始输入 其输出即为input

    cloud_vertex = "cloud"  # 云端设备节点
    edge_vertex = "edge"  # 边缘设备节点

    print(f"start construct graph for model...")
    graph.add_edge(edge_vertex, "v0", capacity=inf)  # 构建模型初始输入v0
    vertex_index = 0  # 构建图的顶点序号

    for layer_index, layer in enumerate(model):
        # 对于某一层先检查其输入是否要进行修改
        if model.has_dag_topology and (layer_index+1) in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[layer_index+1]  # 取出其前置输入条件
            if isinstance(pre_input_cond, list):  # 如果其是一个列表，代表当前层有多个输入
                input = []
                for pre_index in pre_input_cond:  # 对于concat操作,输入应为一个列表
                    input.append(dict_layer_output[pre_index])
            else:  # 当前层的的输入从其他层或得
                input = dict_layer_output[pre_input_cond]

        # 标记在模型中 record_output_list 中的DNN层需要记录输出
        record_flag = model.has_dag_topology and (layer_index+1) in model.record_output_list
        # 枸橘修改后的input进行边的构建
        vertex_index, input = add_graph_edge(graph, vertex_index, input, layer_index, layer, bandwidth, net_type,
                                             dict_input_size_node_name, dict_node_layer,
                                           dict_layer_input, dict_layer_output, record_flag=record_flag)

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
    len_of_shape = len(input.shape)
    input_shape = str(input.shape)  # 获取当前input的大小

    input_slice = input
    for _ in range(len_of_shape-1):
        input_slice = input_slice[0]
    input_slice = str(input_slice[:3])  # 获取input的前3个数据，保证数据的唯一性

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
