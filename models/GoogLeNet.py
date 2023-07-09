import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any
from collections import abc
import torch.nn.functional as F



class GoogLeNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 init_weights: Optional[bool] = False,
                 blocks: Optional[List[Callable[..., nn.Module]]] = None,
    ) -> None:
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.features = nn.Sequential(
            conv_block(3, 64, in_inception=False,kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2),
            conv_block(64, 64, in_inception=False,kernel_size=1),
            conv_block(64, 192, in_inception=False,kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),
        )

        # inception 3
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)
        self.inception3 = nn.Sequential(
            self.inception3a,
            self.inception3b,
            self.maxpool3,
        )

        # inception 4
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        # self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4c = inception_block(512, 64, 32, 512, 24, 512, 64)
        # origin : 512
        self.inception4d = inception_block(1152, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.inception4 = nn.Sequential(
            self.inception4a,
            self.inception4b,
            self.inception4b,
            self.inception4b,
            self.inception4c,
            self.inception4d,
            self.inception4e,
            self.maxpool4
        )

        # inception 5
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.inception5 = nn.Sequential(
            self.inception5a,
            self.inception5b,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.block_list = [self.features, self.inception3, self.inception4, self.inception5,self.classifier]
        self.accumulate_len = []
        ac_len = 0
        for i in range(len(self.block_list)):
            ac_len += len(self.block_list[i])
            self.accumulate_len.append(ac_len)

    def forward(self, x):
        x = self.features(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.classifier(x)
        return x

    def __len__(self):
        return self.accumulate_len[-1]

    def __iter__(self):
        return SentenceIterator(self.block_list,self.accumulate_len)

    def __getitem__(self, item):
        try:
            if item < self.accumulate_len[0]:
                layer = self.features[item]
            elif item < self.accumulate_len[1]:
                layer = self.inception3[item - self.accumulate_len[0]]
            elif item < self.accumulate_len[2]:
                layer = self.inception4[item - self.accumulate_len[1]]
            elif item < self.accumulate_len[3]:
                layer = self.inception5[item - self.accumulate_len[2]]
            else:
                layer = self.classifier[item - self.accumulate_len[3]]
        except IndexError:
            raise StopIteration()
        return layer


class SentenceIterator(abc.Iterator):
    def __init__(self,block_list,accumulate_len):
        self.block_list = block_list
        self.accumulate_len = accumulate_len

        self._index = 0
    def __next__(self):
        try:
            if self._index < self.accumulate_len[0]:
                layer = self.block_list[0][self._index]
            elif self._index < self.accumulate_len[1]:
                layer = self.block_list[1][self._index - self.accumulate_len[0]]
            elif self._index < self.accumulate_len[2]:
                layer = self.block_list[2][self._index - self.accumulate_len[1]]
            elif self._index < self.accumulate_len[3]:
                layer = self.block_list[3][self._index - self.accumulate_len[2]]
            else:
                layer = self.block_list[4][self._index - self.accumulate_len[3]]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer




class BasicConv2d(torch.nn.Sequential):
    """
        conv_block = blocks[0]  BasicConv2d
    """
    def __init__(self,in_channels: int, out_channels: int,in_inception=True,**kwargs: Any) -> None:
        # in_inception = False
        if in_inception:
            half_channels = int(in_channels * 0.25)
            conv = nn.Conv2d(in_channels, half_channels, bias=False, **kwargs)
            conv2 = nn.Conv2d(half_channels, out_channels, bias=False, kernel_size=(1, 1))
            bn = nn.BatchNorm2d(out_channels, eps=0.001)
            layers = [conv, conv2, bn]
        else:
            conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            bn = nn.BatchNorm2d(out_channels, eps=0.001)
            layers = [conv, bn]

        super(BasicConv2d, self).__init__(*layers)




class Operation_Concat(nn.Module):
    """
    拼接操作用于 Google net 最后的拼接操作
    """
    def __init__(self):
        super().__init__()
        self.res = 0
    def forward(self,outputs):
        self.res = torch.cat(outputs,1)
        return self.res

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



class Inception(nn.Module):
    """
    inception_block = blocks[1] Inception
    """
    def __init__(self,
                 in_channels:int,ch1x1: int,ch3x3red: int,
                 ch3x3: int,ch5x5red: int,ch5x5: int,
                 pool_proj: int,conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = nn.Sequential(
            conv_block(in_channels, ch1x1, kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )
        self.concat = Operation_Concat()
        self.branch_list = [self.branch1, self.branch2, self.branch3, self.branch4]

        self.accumulate_len = []
        ac_len = 0
        for i in range(4):
            ac_len += len(self.branch_list[i])
            self.accumulate_len.append(ac_len)


    def _forward(self,x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return self.concat(outputs)

    def __len__(self):
        return self.accumulate_len[-1] + 1

    def __getitem__(self, item):
        try:
            if item < self.accumulate_len[0]:
                layer = self.branch_list[0][item]
            elif item < self.accumulate_len[1]:
                layer = self.branch_list[1][item - self.accumulate_len[0]]
            elif item < self.accumulate_len[2]:
                layer = self.branch_list[2][item - self.accumulate_len[1]]
            elif item < self.accumulate_len[3]:
                layer = self.branch_list[3][item - self.accumulate_len[2]]
            elif item == self.accumulate_len[3]:
                layer = self.concat
            else:
                raise StopIteration()
        except IndexError:
            raise StopIteration()
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
        try:
            if self._index < self.accumulate_len[0]:
                layer = self.branch_list[0][self._index]
            elif self._index < self.accumulate_len[1]:
                layer = self.branch_list[1][self._index - self.accumulate_len[0]]
            elif self._index < self.accumulate_len[2]:
                layer = self.branch_list[2][self._index - self.accumulate_len[1]]
            elif self._index < self.accumulate_len[3]:
                layer = self.branch_list[3][self._index - self.accumulate_len[2]]
            elif self._index == self.accumulate_len[3]:
                layer = self.concat
            else:
                raise StopIteration()
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer



class InceptionAux(nn.Module):
    """
     inception_aux_block = blocks[2] InceptionAux
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x : Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

