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
        self.preInference = nn.Conv2d(in_channels=in_channels,out_channels=192,kernel_size=(7,7), stride=(2,2), padding=3)
        self.up_conv = nn.Conv2d(in_channels=192,out_channels=32,kernel_size=(3,3),padding=1)
        self.down_conv = nn.Conv2d(in_channels=192,out_channels=16,kernel_size=(3,3),padding=1)
        self.concat = Operation_Concat()

        self.layer_list = [self.preInference, self.up_conv, self.down_conv, self.concat]

        # 如果是DAG拓扑结构需要自己设计好下面几个设定
        self.has_dag_topology = True
        self.record_output_list = [1,2,3]  # 哪几层需要保存输出
        self.dag_dict = {   # 定义DAG拓扑相关层的输入
            2: 1,
            3: 1,
            4: [2, 3],
        }

    def _forward(self,x: Tensor) -> List[Tensor]:
        x = self.preInference(x)
        x1 = self.up_conv(x)
        x2 = self.down_conv(x)
        outputs = [x1,x2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.concat(outputs)

    def __len__(self):
        return self.accumulate_len[-1] + 1

    def __getitem__(self, item):
        # 如果超出范围 则停止迭代
        if self.item >= len(self.layer_list):
            raise StopIteration()

        # 根据传入的item取出正确的DNN层
        layer = self.layer_list[item]
        return layer

    def __iter__(self):
        return Inception_SentenceIterator(self.layer_list)



class Inception_SentenceIterator(abc.Iterator):
    def __init__(self,layer_list):
        self.layer_list = layer_list
        self._index = 0

    def __next__(self):
        # 如果超出范围 则停止迭代
        if self._index >= len(self.layer_list):
            raise StopIteration()

        # 根据传入的item取出正确的DNN层
        layer = self.layer_list[self._index]
        self._index += 1
        return layer

