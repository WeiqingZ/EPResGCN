import pandas as pd
import numpy as np
import os
import scipy
import scipy.interpolate as interpolate
import networkx as nx


def one_hot_encode(types, number_class=4):
    one_hot = np.zeros((types.shape[0], number_class))
    for i in range(one_hot.shape[0]):
        one_hot[i, int(types[i])] = 1
    return one_hot


def get_boundary_mask(points, edges_index):
    boundary_mask = np.zeros((len(points), 1))
    # find inlet
    inlet_index = np.setdiff1d(edges_index[0, :], edges_index[1, :]).reshape(-1, 1).astype(int)
    # find outlet
    outlet_mask = np.setdiff1d(edges_index[1, :], edges_index[0, :]).reshape(-1, 1).astype(int)
    # create boundary_mask
    boundary_mask[inlet_index] = 1
    boundary_mask[outlet_mask] = 2
    return boundary_mask


def cul_normal_distance(points, edges):
    distances = []
    for i in range((edges.shape[1])):
        distance = np.linalg.norm(points[edges[0, i]].reshape(1, -1) - points[edges[1, i]].reshape(1, -1))
        distances.append(distance)
    edges_attr = np.array(distances).reshape(-1, 1)

    return edges_attr


def data_enforce(label, windows, stride):
    num_windows = (label.shape[1] - windows) // stride + 1
    labels = []

    for i in range(num_windows):
        start = i * stride
        end = start + windows
        window = label[:, start:end, :]
        labels.append(window)
    return labels


def bid_attr(edges_index, type, attr):
    edges_index = np.concatenate((edges_index, np.flipud(edges_index)), axis=1)
    type = np.concatenate((type, type), axis=0)
    attr = np.concatenate((attr, attr), axis=0)

    return edges_index, type, attr


def cul_edges_attr(points, edges_pro, edges_attr, branch):
    edges_begin = []
    edges_end = []
    # directed edges
    for i in range(1, len(edges_pro)):
        edges_begin.append(edges_pro[i])
        edges_end.append(i)

    boundary_mask = np.zeros((len(points), 1))
    edges_index = np.concatenate((np.array(edges_begin).reshape(1, -1), np.array(edges_end).reshape(1, -1)), axis=0).astype(int)
    e_type = np.zeros((edges_index.shape[1], 1))  # 节点之间的连接设置为 0
    outlet = np.setdiff1d(edges_index[1, :], edges_index[0, :]).reshape(-1, 1).astype(int)
    inlet = np.setdiff1d(edges_index[0, :], edges_index[1, :]).reshape(-1, 1).astype(int)
    boundary_mask[inlet] = 1
    boundary_mask[outlet] = 2
    all_let = np.concatenate((inlet, outlet), axis=0)

    graph = nx.Graph()
    for i in range(edges_index.shape[1]):
        start_node = edges_index[0, i]
        end_node = edges_index[1, i]
        wight = edges_attr[i]
        graph.add_edge(start_node, end_node, wight=wight)

    # 建立连接出口的边,选择每个点最近的出口，入口和出口不作为起点
    b_edge_begin = []
    b_edge_end = []
    b_e_type = []
    b_e_attr = []

    in_edge_begin = []
    in_edge_out = []
    in_edge_type = []
    in_edge_attr = []
    for i in range(points.shape[0]):
        if i in outlet or i in inlet:
            continue
        min_len = float('inf')
        e_out = i
        for out in all_let:
            out = int(out)
            paths = nx.shortest_path(graph, source=i, target=out, weight='wight')  # 获取最短路径的path
            lengths = 0
            for j in range(len(paths) - 1):  # 计算最短路径的长度
                start_node = paths[j]
                end_node = paths[j + 1]
                weight = graph[start_node][end_node]['wight']
                lengths += weight
            if lengths < min_len:  # 选择离源节点最近的出口节点进行链接
                min_len = lengths  # 更新长度
                e_out = out  # 更新出口节点
        if e_out in inlet:
            in_edge_begin.append(0)
            in_edge_out.append(i)
            in_edge_type.append(1)
            in_edge_attr.append(min_len)
        if e_out in outlet:
            b_edge_begin.append(i)
            b_e_type.append(2)
            b_edge_end.append(e_out)
            b_e_attr.append(min_len)

    b_edges_index = np.concatenate((np.array(b_edge_begin).reshape(1, -1), np.array(b_edge_end).reshape(1, -1)),
                                   axis=0).astype(int)
    b_e_type = np.array(b_e_type).reshape(-1, 1)
    b_e_attr = np.array(b_e_attr).reshape(-1, 1)

    in_edges_index = np.concatenate((np.array(in_edge_begin).reshape(1, -1), np.array(in_edge_out).reshape(1, -1)),
                                    axis=0).astype(int)
    in_edge_type = np.array(in_edge_type).reshape(-1, 1)
    in_edge_attr = np.array(in_edge_attr).reshape(-1, 1)

    # 建立分支血管入口到分支内部的边
    branch_num = np.max(branch).astype(int)
    branch_begin = []
    branch_end = []
    branch_type = []
    branch_attr = []
    for i in range(branch_num + 1):
        branch_index = np.where(branch == i)[0]
        for j in branch_index[1:]:
            branch_type.append(3)
            branch_begin.append(branch_index[0])
            branch_end.append(j)
            paths = nx.shortest_path(graph, source=i, target=j, weight='wight')
            lengths = 0
            for x in range(len(paths) - 1):  # 计算路径的长度
                start_node = paths[x]
                end_node = paths[x + 1]
                weight = graph[start_node][end_node]['wight']
                lengths += weight
            branch_attr.append(lengths)

    branch_index = np.concatenate((np.array(branch_begin).reshape(1, -1), np.array(branch_end).reshape(1, -1)),
                                   axis=0).astype(int)
    branch_type = np.array(branch_type).reshape(-1, 1)
    branch_attr = np.array(branch_attr).reshape(-1, 1)

    # 整合所有属性
    edges_index, e_type, edges_attr = bid_attr(edges_index, e_type, edges_attr)
    b_edges_index, b_e_type, b_e_attr = bid_attr(b_edges_index, b_e_type, b_e_attr)
    in_edges_index, in_edge_type, in_edge_attr = bid_attr(in_edges_index, in_edge_type, in_edge_attr)
    branch_index, branch_type, branch_attr = bid_attr(branch_index, branch_type, branch_attr)

    index = np.concatenate((edges_index, b_edges_index, in_edges_index, branch_index), axis=1)
    type = np.concatenate((e_type, b_e_type, in_edge_type, branch_type), axis=0)
    attr = np.concatenate((edges_attr, b_e_attr, in_edge_attr, branch_attr), axis=0)

    # index = np.concatenate((edges_index, b_edges_index, branch_index), axis=1)
    # type = np.concatenate((e_type, b_e_type, branch_type), axis=0)
    # attr = np.concatenate((edges_attr, b_e_attr, branch_attr), axis=0)
    return index, type, attr, boundary_mask


def get_data(file):
    data = pd.read_csv(file).values
    points_number = len(np.where(data[:, 0] == 0.005)[0])
    coordinate = data[:, 1:4][0:points_number, :]
    point = data[:, 4:5][0:points_number, :]
    rcr = data[:, 9:11][0:points_number, :]
    p_type = data[:, 5:9][0:points_number, :]
    edge = data[:, 11:12][0:points_number, :]
    edges_attr = data[:, 12:13][1:points_number, :] / 10
    label = data[:, 13:15].reshape(-1, points_number, 2)
    label = np.transpose(label, (1, 0, 2)).copy()
    time = data[:, 0:1].reshape(-1, points_number)
    time = np.transpose(time, (1, 0)).copy()
    branch = data[:, 15:16][0:points_number, :]
    points = []
    edges_indexs = []
    boundary_masks = []
    edges_attrs = []
    labels = []
    times = []
    rcrs = []
    p_types = []
    e_types = []
    coordinates = []
    edges = []
    branchs = []

    edges_index, edges_types, edges_attr, boundary_mask = cul_edges_attr(point, edge, edges_attr, branch)
    edges_types = one_hot_encode(edges_types, number_class=4)

    for i in range(label.shape[1]):
        points.append(point)
        boundary_masks.append(boundary_mask)
        edges_attrs.append(edges_attr)
        edges_indexs.append(edges_index)
        labels.append(label[:, i, :])
        times.append(time[:, i: i+1])
        rcrs.append(rcr)
        p_types.append(p_type)
        e_types.append(edges_types)
        coordinates.append(coordinate)
        edges.append(edge)
        branchs.append(branch)

    return points, edges_indexs, boundary_masks, edges_attrs, labels, times, rcrs, p_types, e_types, coordinates, edges, branchs



