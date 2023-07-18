import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Callable, Any
from collections import abc


def getBlockIndex(item, accumulate_len):
    """
    通过传入的下标item，提供改item代表的层应该在哪个模块中选择
    :param item: item or index 层的下标，从0开始计数
    :param accumulate_len: 代表各部分累加和的列表
    :return: 对应的模块下标 part_index part_index = 0 代表features 以此类推 part_index = 1 代表inception3
    """
    for part_index in range(len(accumulate_len)):
        part_len = accumulate_len[part_index]
        # 找到属于哪个模块
        if item < part_len:
            return part_index
    return len(accumulate_len)


class Operation_Concat(nn.Module):
    """
    Operation_Concat 用于最后的拼接操作
    """
    def __init__(self):
        super().__init__()
        self.res = 0
    def forward(self,outputs):
        self.res = torch.cat(outputs,1)
        return self.res


class EasyModel(nn.Module):
    """
    构建一个InceptionBlock结构 ： 是一个DAG拓扑结构的模型
    """
    def __init__(self,in_channels:int = 3) -> None:
        super(EasyModel, self).__init__()
        self.preInference = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(7, 7), stride=(2, 2))
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1)
        )
        self.concat = Operation_Concat()

        self.branch_list = [self.preInference, self.branch1, self.branch2]
        self.accumulate_len = []
        for i in range(len(self.branch_list)):
            if i == 0:
                self.accumulate_len.append(len(self.branch_list[i]))
            else:
                self.accumulate_len.append(self.accumulate_len[i - 1] + len(self.branch_list[i]))


        # 如果是DAG拓扑结构需要自己设计好下面几个设定
        self.has_dag_topology = True
        self.record_output_list = [self.accumulate_len[0], self.accumulate_len[1], self.accumulate_len[2]]  # 哪几层需要保存输出
        self.dag_dict = {   # 定义DAG拓扑相关层的输入
            self.accumulate_len[0] + 1: self.accumulate_len[0],
            self.accumulate_len[1] + 1: self.accumulate_len[0],
            self.accumulate_len[2] + 1: [self.accumulate_len[1], self.accumulate_len[2],],
        }

    def _forward(self,x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1,branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        x = self.preInference(x)
        outputs = self._forward(x)
        return self.concat(outputs)

    def __len__(self):
        return self.accumulate_len[-1] + 1

    def __getitem__(self, item):
        # 如果超出范围 则停止迭代
        if item >= self.accumulate_len[-1] + 1:
            raise StopIteration()

        # 根据传入的item取出正确的DNN层
        part_index = getBlockIndex(item, self.accumulate_len)
        if part_index == 0:
            layer = self.branch_list[part_index][item]
        elif part_index < len(self.accumulate_len):
            layer = self.branch_list[part_index][item - self.accumulate_len[part_index - 1]]
        else:
            layer = self.concat
        return layer

    def __iter__(self):
        return Inception_SentenceIterator(self.branch_list,self.concat,self.accumulate_len)



class Inception_SentenceIterator(abc.Iterator):
    def __init__(self,branch_list,concat,accumulate_len):
        self.branch_list = branch_list
        self.accumulate_len = accumulate_len
        self.concat = concat

        self._index = 0


    def __next__(self):
        # 如果超出范围 则停止迭代
        if self._index >= self.accumulate_len[-1] + 1:
            raise StopIteration()

        # 根据传入的item取出正确的DNN层
        part_index = getBlockIndex(self._index, self.accumulate_len)
        if part_index == 0:
            layer = self.branch_list[part_index][self._index]
        elif part_index < len(self.accumulate_len):
            layer = self.branch_list[part_index][self._index - self.accumulate_len[part_index - 1]]
        else:
            layer = self.concat

        self._index += 1
        return layer


class easy_dag_part(nn.Module):
    def __init__(self,branches):
        super(easy_dag_part, self).__init__()
        self.branch1 = branches[0]
        self.branch2 = branches[1]
        self.concat = Operation_Concat()
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1, branch2]
        return self.concat(outputs)


class EdgeInception(nn.Module):
    """
    edge Inception 用于构建划分好的边端Inception
    """
    def __init__(self,edge_branches):
        super(EdgeInception, self).__init__()
        self.branch1 = edge_branches[0]
        self.branch2 = edge_branches[1]
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1, branch2]
        return outputs


class CloudInception(nn.Module):
    """
        cloud Inception 用于构建划分好的云端Inception
    """
    def __init__(self, cloud_branches):
        super(CloudInception, self).__init__()
        self.branch1 = cloud_branches[0]
        self.branch2 = cloud_branches[1]
        self.concat = Operation_Concat()

    def forward(self, x):
        branch1 = self.branch1(x[0])
        branch2 = self.branch2(x[1])
        outputs = [branch1, branch2]
        return self.concat(outputs)


def construct_edge_cloud_inception_block(model: EasyModel, model_partition_edge: list):
    """
    构建Inception的边端模型和云端模型
    :param model: 传入一个需要划分的Inception block
    :param model_partition_edge: Inception的划分点 (start_layer,end_layer)
    :return: edge_Inception,cloud_Inception
    """
    accumulate_len = model.accumulate_len
    edge_model,cloud_model = nn.Sequential(),nn.Sequential()
    if len(model_partition_edge) == 1:  # 只有一个地方需要划分
        partition_point = model_partition_edge[0][0]
        assert partition_point <= accumulate_len[0] + 1
        idx = 1
        for layer in model:
            if idx > accumulate_len[0]: break
            if idx <= partition_point:
                edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            else:
                cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            idx += 1
        layer = easy_dag_part(model.branch_list[1:])
        cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
    else:  # 需要在4个branch之间进行划分
        assert len(model_partition_edge) == 2
        branches = model.branch_list[1:]
        edge_model.add_module(f"1-preInference", model.preInference)

        edge_branches = []
        cloud_branches = []
        for edge in model_partition_edge:
            edge_branch = nn.Sequential()
            cloud_branch = nn.Sequential()

            block,tmp_point = None,None
            if edge[0] in range(accumulate_len[0] + 1, accumulate_len[1] + 1) or edge[1] in range(accumulate_len[0] + 1,accumulate_len[1] + 1):
                block = branches[0]
                tmp_point = edge[0] - accumulate_len[0]
            elif edge[0] in range(accumulate_len[1] + 1, accumulate_len[2] + 1) or edge[1] in range(accumulate_len[1] + 1, accumulate_len[2] + 1):
                block = branches[1]
                tmp_point = edge[0] - accumulate_len[1]

            idx = 1
            for layer in block:
                if idx <= tmp_point:
                    edge_branch.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                else:
                    cloud_branch.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                idx += 1

            edge_branches.append(edge_branch)
            cloud_branches.append(cloud_branch)

        # 使用 edge_branches 以及 cloud_branches 构建 EdgeInception 以及 CloudInception 两个类
        edge_Inception = EdgeInception(edge_branches)
        cloud_Inception = CloudInception(cloud_branches)

        edge_model.add_module(f"2-edge-inception", edge_Inception)
        cloud_model.add_module(f"1-cloud-inception", cloud_Inception)
    return edge_model, cloud_model
