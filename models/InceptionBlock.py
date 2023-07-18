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


class BasicConv2d(torch.nn.Sequential):
    """
        一个简单的块结构, conv+bn+
    """
    def __init__(self,in_channels: int, out_channels: int, **kwargs: Any) -> None:
        conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        bn = nn.BatchNorm2d(out_channels, eps=0.001)
        layers = [conv, bn]
        super(BasicConv2d, self).__init__(*layers)


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


class InceptionBlock(nn.Module):
    """
    构建一个InceptionBlock结构 ： 是一个DAG拓扑结构的模型
    """
    def __init__(self,
                 in_channels:int = 3,ch1x1: int = 64,ch3x3red: int = 96,
                 ch3x3: int = 128,ch5x5red: int = 16,ch5x5: int = 32,
                 pool_proj: int = 32,conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(InceptionBlock, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.preInference = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2),
            conv_block(64, 64, kernel_size=1),
            conv_block(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),
        )
        out_pre_channels = 192

        self.branch1 = nn.Sequential(
            conv_block(out_pre_channels, ch1x1, kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            conv_block(out_pre_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(out_pre_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(out_pre_channels, pool_proj, kernel_size=1),
        )
        # concat帮助实现拼接操作
        self.concat = Operation_Concat()
        self.branch_list = [self.preInference, self.branch1, self.branch2, self.branch3, self.branch4]

        # inception各部分的累加和 用于inception内部的遍历
        self.accumulate_len = []
        for i in range(len(self.branch_list)):
            if i == 0:
                self.accumulate_len.append(len(self.branch_list[i]))
            else:
                self.accumulate_len.append(self.accumulate_len[i-1] + len(self.branch_list[i]))

        # 如果是DAG拓扑结构需要自己设计好下面几个设定
        self.has_dag_topology = True
        self.record_output_list = [self.accumulate_len[0], self.accumulate_len[1], self.accumulate_len[2],
                                   self.accumulate_len[3], self.accumulate_len[4]]  # 哪几层需要保存输出
        self.dag_dict = {  # 定义DAG拓扑相关层的输入
            self.accumulate_len[0] + 1: self.accumulate_len[0],
            self.accumulate_len[1] + 1: self.accumulate_len[0],
            self.accumulate_len[2] + 1: self.accumulate_len[0],
            self.accumulate_len[3] + 1: self.accumulate_len[0],
            self.accumulate_len[4] + 1: [self.accumulate_len[1], self.accumulate_len[2],
                                         self.accumulate_len[3], self.accumulate_len[4],]
        }


    def _forward(self,x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
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


class inception_dag_part(nn.Module):
    """
        提取出inception中的DAG部分 设self.preInference层数为p
        则在第p层（包含第p层）之后进行划分 可以将后面的部分直接使用inception_dag_part
    """
    def __init__(self,branches):
        super(inception_dag_part, self).__init__()
        self.branch1 = branches[0]
        self.branch2 = branches[1]
        self.branch3 = branches[2]
        self.branch4 = branches[3]
        self.concat = Operation_Concat()
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return self.concat(outputs)


class EdgeInception(nn.Module):
    """
    edge Inception 用于构建划分好的边端Inception
    """
    def __init__(self,edge_branches):
        super(EdgeInception, self).__init__()
        self.branch1 = edge_branches[0]
        self.branch2 = edge_branches[1]
        self.branch3 = edge_branches[2]
        self.branch4 = edge_branches[3]
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs


class CloudInception(nn.Module):
    """
        cloud Inception 用于构建划分好的云端Inception
    """
    def __init__(self, cloud_branches):
        super(CloudInception, self).__init__()
        self.branch1 = cloud_branches[0]
        self.branch2 = cloud_branches[1]
        self.branch3 = cloud_branches[2]
        self.branch4 = cloud_branches[3]
        self.concat = Operation_Concat()

    def forward(self, x):
        branch1 = self.branch1(x[0])
        branch2 = self.branch2(x[1])
        branch3 = self.branch3(x[2])
        branch4 = self.branch4(x[3])

        outputs = [branch1, branch2, branch3, branch4]
        return self.concat(outputs)


def construct_edge_cloud_inception_block(model: InceptionBlock, model_partition_edge: list):
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
        layer = inception_dag_part(model.branch_list[1:])
        cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
    else:  # 需要在4个branch之间进行划分
        assert len(model_partition_edge) == 4
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
            elif edge[0] in range(accumulate_len[2] + 1, accumulate_len[3] + 1) or edge[1] in range(accumulate_len[2] + 1, accumulate_len[3] + 1):
                block = branches[2]
                tmp_point = edge[0] - accumulate_len[2]
            elif edge[0] in range(accumulate_len[3] + 1, accumulate_len[4] + 1) or edge[1] in range(accumulate_len[3] + 1, accumulate_len[4] + 1):
                block = branches[3]
                tmp_point = edge[0] - accumulate_len[3]

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