import networkx as nx
import sys
import torch
import DAG_utils
sys.path.append("../DNN_Architecture")
sys.path.append("../work1_Neuronsurgeon")
sys.path.append("../work2_Edgent")
import function
from collections.abc import Iterable
import torch.nn as nn
import ResNet
import ResNet2
import GoogLeNet
import GoogLeNet2
import Inceptionv2

from MobileNet import InvertedResidual
def test_MobileNet_InvertedResidual():
    """
    测试block : MobileNet - InvertedResidual、ConvNormActivation
    :return:
    """
    x = torch.rand((1, 64, 56, 56))
    # InvertedResidual_block = InvertedResidual(64, 16, stride=1, expand_ratio=1)
    InvertedResidual_block = InvertedResidual(64, 128, stride=1, expand_ratio=0.5)
    InvertedResidual_block.eval()

    print(InvertedResidual_block)

    partition, cut_value = DAG_utils.get_block_value(InvertedResidual_block, x, show=True)
    reachable, non_reachable = partition

    print(reachable)
    print(non_reachable)
    print(cut_value)
    print("============================")




def test_resnet_BasicBlock():
    in_planes = 64
    planes = 64
    stride = 1
    down_sample = nn.Sequential(
        nn.Conv2d(in_planes, planes,kernel_size=(1,1),stride=(stride,stride)),
        nn.BatchNorm2d(planes)
    )

    block = ResNet.BasicBlock(in_planes, planes, stride, down_sample, nn.BatchNorm2d)

    x = torch.rand((1,64,56,56))
    block =function.getDnnModel(4)[4]
    print(block)

    DAG_utils.get_resnet_block_value(block,x,show = True)


def test_google_Inception():
    x = torch.rand((1,192,30,30))
    inception = GoogLeNet.Inception(192, 16, 16, 16, 16, 32, 32)

    graph,dict_vertex_layer = DAG_utils.get_google_inception_block_value(inception,x,show=False,getgraph=True)
    start,end = 'edge','cloud'

    # 显示从哪里划分
    print("=============================")
    _,_,cut_layer_list = DAG_utils.get_partition_point(graph,start,end,dict_vertex_layer)
    DAG_utils.show_partition_layer(inception,cut_layer_list)

    # print(DAG_utils.get_transmission_lat(x, network_type=3))


def recursion_partition_dnn(model,cut_layer_list,dict_layer_input,now_depth,limit_depth,dads=False):
    """
    通过递归函数 构建一个多层划分策略
    多层策略使用字典表示 每一层表示如何对下一层进行划分
    :param model: 传入需要进行分层的DNN模型
    :param cut_layer_list: 表示从dnn模型的第几层进行划分
    :param dict_layer_input: 可以根据dnn模型的层数 - 获取对应层数的dnn layer需要的输入
    :return:表示多层划分策略的字典 dict
    """
    if dads and now_depth >= limit_depth:
        return {}

    if isinstance(model,Iterable):
        model_len = len(model)
        block_dict = {}
        # 如果 cut_layer_list 是[],代表分割点全在v - cloud,代表的是 edge-only 的情况
        # dnn的所有层都在边端端设备上运行 返回策略为 (len(model),len(model) + 1)
        if len(cut_layer_list) == 0:
            block_dict[(model_len, model_len + 1)] = {}
            return block_dict

        for cut_edge in cut_layer_list:
            # 取出在第几层之后对模型进行划分
            start_layer_index = cut_edge[0]
            end_layer_index = cut_edge[1]

            # 代表block不可迭代，不可以递归寻找模型划分点
            # {} 代表此结点指向的block不可迭代 或者 此结点指向的block不需要进行划分,全放在边端执行情况:block 返回 (0,0)
            if start_layer_index == end_layer_index:  # 处理类似于 (0,0) (5,5)的情况
                if start_layer_index == 0:  # (0,0)的划分情况 不用划分 直接全部dnn模型由cloud端运行
                    block_dict[cut_edge] = {}
                    continue
                else:  # (5,5)的划分情况 不用划分 将第5层(包括第五层)之前的所有dnn放到边端执行 之后的部分放到云端执行
                    block_dict[cut_edge] = {}
                    continue
            else:  # 处理类似与 (0,1) (7,8) 的情况；需要对第1层block 或者 第8层block进行进一步划分
                # function.model_partition 表示在第index层之后对模型进行划分
                block = model[end_layer_index - 1]
                # print(end_layer_index)
                # print(block)

                if not isinstance(block, Iterable):
                    block_dict[cut_edge] = {}
                    # print("layer is not instance of Iterable")
                else:
                    # 第 end_layer_index 层的输入，end_layer_index = 1 代表第一层的输入
                    x = dict_layer_input[end_layer_index]
                    # 递归调用函数
                    graph, new_dict_vertex_layer,new_dict_layer_input = DAG_utils.get_block_value(block, x, show=False,getgraph=True,dads=dads)
                    _, _, new_cut_layer_list = DAG_utils.get_partition_point(graph, 'edge', 'cloud', new_dict_vertex_layer)
                    # print(new_cut_layer_list)

                    res_dict = recursion_partition_dnn(block,new_cut_layer_list,new_dict_layer_input,now_depth+1,limit_depth,dads=dads)
                    block_dict[cut_edge] = res_dict
        return block_dict
    else:  # block 不可迭代，为单层的layer结构
        return {}



def get_partition_model(model,block_dict,edge_model,cloud_model,index):
    """
    即runtime阶段措施
    通过得到的 block dict，划分model为 edge model 以及 cloud model
    实现 DAPF 对于 DNN model 的adaptive partition
    :param model: 需要分割的模型
    :param block_dict: 记录分割点的字典
    :param edge_model: 边端执行模型
    :param cloud_model: 云端执行模型
    :param index: 记录model layer的下标
    :return:
    """
    # 字典为空说明不需要进一步划分 直接返回即可
    if block_dict == {}:
        cloud_model.add_module(f"{index}-{model.__class__.__name__}",model)
        index += 1
        return index

    keys = block_dict.keys()
    if (isinstance(model,GoogLeNet.Inception) or isinstance(model,GoogLeNet2.Inception)
        or isinstance(model,Inceptionv2.Inception)
        ) and len(keys) > 1:
        edge_Inception,cloud_Inception = construct_edge_inception(model,block_dict)
        edge_model.add_module(f"{index}-{edge_Inception.__class__.__name__}", edge_Inception)
        index += 1
        cloud_model.add_module(f"{index}-{cloud_Inception.__class__.__name__}", cloud_Inception)
        index += 1
        return index


    assert len(keys) == 1
    # 获得模型分割点
    edge = list(keys)[0]
    partition_point = edge[1] - 1 if edge[0] != edge[1] else edge[1]

    # 将对应的layer添加到边端模型和云端模型中
    index = construct_edge_model(model,edge_model,partition_point,index)

    # start layer == end layer的情况不用对下一层进行划分
    # 处理类似于 (0,0) (5,5)的情况
    # (0,0)的划分情况 不用划分 直接全部dnn模型由cloud端运行
    # (5,5)的划分情况 不用划分 将第5层(包括第五层)之前的所有dnn放到边端执行 之后的部分放到云端执行
    if edge[0] == edge[1]:
        if partition_point == 0 and (isinstance(model,ResNet.BasicBlock) or isinstance(model,ResNet2.BasicBlock)):
            cloud_model.add_module(f"{index}-resnet-BasicBlock", model)
            index += 1
        else:
            index = construct_cloud_model(model,cloud_model,partition_point,index)
        return index
    elif partition_point < len(model):
        child_dict = block_dict[edge]
        # 取出模型的第 partition_point + 1 层
        block = model[partition_point]
        # 递归对child block进行模型分割
        # print(block)
        # print("-----------------",partition_point)
        index = get_partition_model(block,child_dict,edge_model,cloud_model,index)

    # partition_point = len(model)的话即所有dnn layer都放到边端执行
    index = construct_cloud_model(model, cloud_model, partition_point+1, index)

    # print(edge_model)
    # print("-- - - - - - -- - - - - -- - -- -- - -- -- - ")
    # print(cloud_model)
    return index



def construct_edge_model(model,edge_model,partition_point,index):
    """
    partition point = 7 , 则代表第七层
    model[partition = 7]  代表第八层

    第 partition point 层之前的layer(包括partition point)放入到edge model中
    partition point + 1 位置的layer用于递归分割
    partition point + 1 之后的layer放入到cloud model中
    构建云端运行模型 用于在云端执行dnn
    :param model: 原需要分割的模型
    :param edge_model: 构建过程中的边端模型
    :param partition_point: 模型分割点
    :param index: 当前下标
    """
    idx = 1
    for layer in model:
        if idx <= partition_point:
            edge_model.add_module(f"{index}-{layer.__class__.__name__}", layer)
            index += 1
            idx += 1
        else:
            idx += 1
            continue
    return index



def construct_cloud_model(model,cloud_model,partition_point,index):
    """
    partition point之前的layer(包括partition point)放入到edge model中
    partition point + 1 位置的layer用于递归分割
    partition point + 1 之后的layer放入到cloud model中
    构建云端运行模型 用于在云端执行dnn
    :param model: 原需要分割的模型
    :param cloud_model: 构建过程中的云端模型
    :param partition_point: 模型分割点
    :param index: 当前下标
    """
    idx = 1
    for layer in model:
        if idx <= partition_point:
            idx += 1
            continue
        else:
            cloud_model.add_module(f"{index}-{layer.__class__.__name__}", layer)
            index += 1
            idx += 1
    return index



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




def get_model_min_cut_partition(model,now_depth,limit_depth,show=False,dads=False,get_partition_point=False,net_type=3,define_speed=56):
    """
    测试六种不同DNN架构的 min-cut 最小值以及划分策略
    返回划分之后的edge model 以及 cloud model
    """
    x = torch.rand((1, 3, 224, 224))
    # x = torch.rand((1, 64, 28, 28))
    # print(net_type)

    if not dads:
        graph, dict_vertex_layer, dict_layer_input = DAG_utils.construct_DAG_by_model(model, x, show=show,net_type=net_type,define_speed=define_speed)
    else:
        graph, dict_vertex_layer, dict_layer_input = DAG_utils.construct_DAG_by_model(model, x, show=show,dads=True,net_type=net_type,define_speed=define_speed)
    # for edge in graph.edges.data():
    #     print(edge)

    # print("顶点个数：", graph.number_of_nodes())
    # print("边的个数：", graph.number_of_edges())
    start = 'edge'
    end = 'cloud'

    cut_value, partition = nx.minimum_cut(graph, start, end)
    reachable, non_reachable = partition
    print(dict_vertex_layer)
    print(f"reachable vertex : {reachable}")
    print(f"non-reachable vertex : {non_reachable}")
    print("================================================")

    cut_set, partition_edge, cut_layer_list = DAG_utils.get_partition_point(graph, start, end, dict_vertex_layer)
    print(f"cut set : {cut_set}")
    print(f"dnn partition edge set : {partition_edge}")

    cut_set_sum = round(sum(graph.edges[u, v]["capacity"] for (u, v) in cut_set),3)
    print(f"min cut value : {round(cut_value,3)} ms , cut set sum : {cut_set_sum} ms")
    if round(cut_value,3) == cut_set_sum:
        print("min cut partition is correct!")
    else:
        print("min cut partition is wrong!")

    # 显示从哪里划分
    # DAG_utils.show_partition_layer(model,cut_layer_list)
    print("===========================================================")

    # 用字典递归的方式 说明在block中如何划分
    print(cut_layer_list)
    partition_dict = recursion_partition_dnn(model,cut_layer_list,dict_layer_input,now_depth,limit_depth,dads=dads)
    print(partition_dict)
    if get_partition_point:
        return str(partition_dict)

    # 获得边端模型和云端模型 用来进行云边协同执行
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    get_partition_model(model,block_dict=partition_dict,edge_model=edge_model,cloud_model=cloud_model,index=1)

    print(edge_model)
    print("=========================== edge - cloud - separate ==========================")
    print(cloud_model)
    print("============================= end partition ==============================")
    print(f"transmit x.shape : {edge_model(x).shape}")
    return edge_model,cloud_model
