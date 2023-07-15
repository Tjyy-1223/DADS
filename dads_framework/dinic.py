import decimal

import networkx as nx
import sys
from collections import deque
from decimal import Decimal

def create_residual_network(origin_digraph):
    """
    根据传入的原始有向图 构建初始化残差网络图
    初始的residual network就是 origin digraph的拷贝
    :param origin_digraph: 原始构建好的有向图
    :return: 构建好的初始残差图 residual_graph
    """
    return origin_digraph.copy()



def bfs_for_level_digraph(residual_digraph):
    """
    根据传入的 residual digraph 使用bfs构建 level digraph
    :param residual_digraph: 残差网络
    :return: 构建好的层级网络信息 level_dict
             以及 最后一个节点是否在dict中(boolean):cloud_node_in_dict , 用于dinic算法终止条件的判断
    """
    level_dict = {}  # 记录节点是否已经被被访问过 同时记录节点的层数
    start_node = 'edge'
    level_dict[start_node] = 1

    # 初始化一个队列 用于bfs遍历
    Q = deque()
    Q.append(start_node)

    # 开始bfs遍历 -> 构建level digraph
    while True:
        if len(Q) == 0:
            break

        # print("-------------")
        node = Q.popleft()  # 弹出上一层次的节点
        # print(f"弹出 : {node}")

        now_level = level_dict[node]
        for neighbor_nodes in nx.neighbors(residual_digraph,node):
            # 如果neighbor_nodes已经在队列里面 就不需要进行重复添加
            if(neighbor_nodes not in level_dict.keys()) and (neighbor_nodes not in Q) \
                    and residual_digraph.get_edge_data(node,neighbor_nodes)["capacity"] > 0:
                level_dict[neighbor_nodes] = now_level + 1
                Q.append(neighbor_nodes)


    # 判断结束节点t是否保存在层级图中
    end_node = 'cloud'
    cloud_node_in_dict = end_node in level_dict.keys()
    return level_dict,cloud_node_in_dict



def dfs_once(residual_graph,level_dict,dfs_start_node,augment_value):
    """
    使用 dfs 方法来不断选取增广路径，一次DFS可以实现多次增广，并在dfs过程中不断修改residual_graph的权重值
    在层次网络中用一次DFS过程进行增广，DFS执行完毕，该阶段的增广也执行完毕。
    :param residual_graph: 残差网络信息
    :param level_dict: 层级网络信息
    :param dfs_start_node: dfs出发点
    :param augment_value: 此次增广的增广值
    :return: 返回增广路径的值
    """
    tmp = augment_value
    end_node = "cloud"

    # 首先排除特殊情况
    if dfs_start_node == end_node:
        return augment_value

    for node in residual_graph.nodes():  # 遍历图中所有顶点
        if level_dict[dfs_start_node] + 1 == level_dict[node]:  # 找到下一层次的节点
            if residual_graph.has_edge(dfs_start_node,node) and residual_graph.get_edge_data(dfs_start_node, node)["capacity"] > 0:  # capacity = 0 表示已经没有容量了 可以不通过这个路径
                capacity = residual_graph.get_edge_data(dfs_start_node, node)["capacity"]
                # print(f"{dfs_start_node} -> {node} : {capacity}")
                # 开始进行dfs找到一个增广路径 并记录增广值（木桶效应 - 取最小值）
                flow_value = dfs_once(residual_graph,level_dict,node,min(tmp,capacity))
                # print(f"flow value :  {flow_value}")

                # 增加反向边 或者 修改反向边的值
                if flow_value > 0:
                    if not residual_graph.has_edge(node,dfs_start_node):
                        residual_graph.add_edge(node, dfs_start_node, capacity=flow_value)
                    else:
                        neg_flow_value = residual_graph.get_edge_data(node,dfs_start_node)["capacity"]
                        residual_graph.add_edge(node, dfs_start_node, capacity=flow_value + neg_flow_value)

                # 处理正向边
                # print(f"{dfs_start_node} -> {node} : {capacity-flow_value}")
                # print("-------------------------------")
                residual_graph.add_edge(dfs_start_node, node, capacity=capacity - flow_value)
                # 如果边权重为0 就可以删除掉这个边了 防止level digraph构建错误
                if capacity - flow_value <= 0:
                    residual_graph.remove_edge(dfs_start_node, node)

                tmp -= flow_value
    return augment_value - tmp


def dinic_algorithm(origin_digraph):
    """
    对有向图使用dinic算法找到 最大流、最小割的解决策略
    :param origin_digraph: 原始构建好的有向图
    :return: min_cut_value, reachable, non_reachable
    """
    min_cut_value = 0
    inf = sys.maxsize

    # 通过原始图创建一个初始的residual digraph
    residual_graph = create_residual_network(origin_digraph)
    # print(residual_graph.edges(data=True))

    for edge in residual_graph.edges(data=True):
        u = edge[0]
        v = edge[1]
        c = Decimal(str(edge[2]['capacity'])).quantize(Decimal('0.000'))
        # print(u,v,c)
        residual_graph.add_edge(u,v,capacity=c)

    # 通过bfs算法构建level dict信息；也可以当成构建level graph
    level_dict, cloud_node_in_dict = bfs_for_level_digraph(residual_graph)
    while cloud_node_in_dict:
        # print("bfs construction")
        # 首先进行一次dfs遍历
        dfs_value = dfs_once(residual_graph,level_dict,dfs_start_node="edge",augment_value=inf)
        min_cut_value += dfs_value
        # print(dfs_value)
        while dfs_value > 0:  # dfs_value > 0 说明还可以继续进行dfs搜索其他增广路径
            # print(residual_graph.edges(data=True))
            # print("dfs search")
            dfs_value = dfs_once(residual_graph, level_dict, dfs_start_node="edge", augment_value=inf)
            min_cut_value += dfs_value

        # 当本阶段dfs遍历结束之后 ，重新生成新的bfs - level digraph进行循环：知道终点不能表示在level digraph中
        level_dict, cloud_node_in_dict = bfs_for_level_digraph(residual_graph)

    # 根据最后的 residual_graph (level_dict)， 从edge可以到达的点属于 reachable，其他顶点属于 non_reachable
    reachable, non_reachable = set(), set()
    for node in residual_graph:
        if node in level_dict.keys():   reachable.add(node)
        else:   non_reachable.add(node)

    return min_cut_value, reachable, non_reachable