import torch.nn as nn
from torch import Tensor
from collections import deque

class DagModel(nn.Module):
    """
        利用传来的 edge_layers , pre_node_dict 以及 suc_node_nums 构建一个新的有向图拓扑模型
    """

    def __init__(self, layers, end_nodes,
                 pre_node_dict, suc_node_dict, dict_node_layer) -> None:
        super(DagModel, self).__init__()
        self.layers = layers
        self.end_nodes = end_nodes
        self.pre_node_dict = pre_node_dict
        self.suc_node_dict = suc_node_dict
        self.dict_node_layer = dict_node_layer

        # 创建一个队列用来依次推理每个顶点代表的DNN层
        self.Q = deque()
        for node in self.layers.keys():
            self.Q.append(node)

        # 创建一个字典用来保存目前已经推理好的layer
        self.res_dict = {}

    def pre_process(self, x):
        if isinstance(x,Tensor):
            self.res_dict["v0"] = x
        elif isinstance(x,dict):
            for key in x.keys():
                self.res_dict[key] = x[key]

    def forward(self, x):
        if len(self.Q) == 0:  # cloud端没有DNN层
            return x

        self.pre_process(x)  # 初始化字典
        # 开始取出node中的节点进行计算
        while True:
            if len(self.Q) == 0: break
            node = self.Q.popleft()  # 弹出上一层次的节点

            pre_nodes = self.pre_node_dict[node]  # 需要的前置节点
            # print(node, pre_nodes, self.res_dict.keys())

            if len(pre_nodes) == 1:  # 只需要一个前置输入
                pre_node = pre_nodes[0]
                if pre_node not in self.res_dict.keys():  # 前置节点还没算法
                    self.Q.append(node)  # 把node节点在放回去
                else:  # 前置条件具备的话就进行推理
                    # 如果前后两个节点指向同一层 不需要进行任何推理
                    if self.dict_node_layer[node] == self.dict_node_layer[pre_node]:
                        self.res_dict[node] = self.res_dict[pre_node]
                    else:
                        layer = self.layers[node]  # 取出对应的层
                        self.res_dict[node] = layer(self.res_dict[pre_node])  # 进行推理并保存结果

                    self.suc_node_dict[pre_node] = self.suc_node_dict[pre_node] - 1
                    # 前一个节点的后续节点数量标记为0 证明不需要保存其结果了
                    if self.suc_node_dict[pre_node] == 0:
                        self.res_dict.pop(pre_node)  # 进行删除
            else:  # 需要多个前置输入的情况
                input_list = []
                for pre_node in pre_nodes:
                    if pre_node not in self.res_dict.keys():
                        break
                    else: input_list.append(self.res_dict[pre_node])
                if len(input_list) == len(pre_nodes):  # 前置节点都已经计算完成
                    layer = self.layers[node]  # 取出对应的层
                    self.res_dict[node] = layer(input_list)  # 进行推理并保存结果
                    for pre_node in pre_nodes:
                        self.suc_node_dict[pre_node] = self.suc_node_dict[pre_node] - 1
                        if self.suc_node_dict[pre_node] == 0:
                            self.res_dict.pop(pre_node)  # 进行删除
                else:
                    self.Q.append(node)  # 把node节点在放回去

        outputs = {}
        for output_node in self.end_nodes:
            outputs[output_node] = self.res_dict[output_node]
        return outputs


    def info(self):
        for node in self.layers.keys():
            print(node,self.layers[node])