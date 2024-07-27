import sys
import os
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch
from graph_cteate.process_csv import *


class MyData(Data):
    def __init__(self, x=None, boundary_condition=None, edge_index=None, edge_attr=None, y=None, coordinate=None,
                 time=None, rcr=None, p_type=None, e_type=None, edge=None, branch=None, y_mean=None, y_std=None):
        super(MyData, self).__init__()
        self.x = x  # 特征矩阵
        self.boundary_mask = boundary_condition
        self.edge_index = edge_index  # 边索引
        self.y = y  # 标签
        self.edge_attr = edge_attr
        self.time = time
        self.rcr = rcr
        self.p_type = p_type
        self.e_type = e_type
        self.coordinate = coordinate
        self.edge = edge
        self.branch = branch
        self.y_mean = y_mean
        self.y_std = y_std

def find_condition(masks, labels):
    conditions = []
    for i in range(len(masks)):
        condition = np.zeros((labels[i].shape[0], 1))
        mask = np.where(masks[i] == 1)[0].reshape(-1, 1)
        condition[mask, 0] = labels[i][mask, 0]
        conditions.append(condition)
    return conditions


def normalize_rcr(datas):
    temp = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in datas], axis=0)
    idx = np.where(np.all(temp != 0, axis=1))[0]
    temp = temp[idx, :]
    mean = np.mean(temp, axis=0)
    std = np.std(temp, axis=0)
    std[std == 0] = 1e-12

    for i in range(len(datas)):
        row_idx = np.where(np.all(datas[i] == 0, axis=1))[0]
        datas[i] = (datas[i] - mean) / std
        datas[i][row_idx] = 0
    return datas, mean, std


def normalize(datas):
    temp = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in datas], axis=0)
    mean = np.mean(temp, axis=0)
    std = np.std(temp, axis=0)
    std[std == 0] = 1e-12
    for i in range(len(datas)):
        datas[i] = (datas[i] - mean) / std
    return datas, mean, std


class MyDataset(InMemoryDataset):
    def __init__(self, root, filepath, transform=None, pre_transform=None):
        self.root_path = root
        self.file_path = filepath
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.file_path)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # 遍历每个文件夹
        data_list = []
        nodes = []
        edge_indexs = []
        attrs = []
        masks = []
        labels = []
        times = []
        p_types = []
        e_types = []
        rcrs = []
        coordinates = []
        edges = []
        branchs = []
        for folder in self.raw_file_names:
            if not folder.endswith(".csv"):
                continue
            folder_path = os.path.join(self.file_path, folder)
            node_data, edge_index, boundary_mask, edge_attr, label, time, rcr, p_type, e_type, coordinate, edge, branch =\
                get_data(folder_path)
            for i in range(len(node_data)):
                nodes.append(node_data[i])
                edge_indexs.append(edge_index[i])
                masks.append(boundary_mask[i])
                attrs.append(edge_attr[i])
                labels.append(label[i])
                times.append(time[i])
                rcrs.append(rcr[i])
                p_types.append(p_type[i])
                e_types.append(e_type[i])
                coordinates.append(coordinate[i])
                edges.append(edge[i])
                branchs.append(branch[i])

        rcrs, _, _ = normalize_rcr(rcrs)
        nodes, x_mean, x_std = normalize(nodes)
        labels, y_mean, y_std = normalize(labels)
        attrs, attr_mean, attr_std = normalize(attrs)
        boundary_conditions = find_condition(masks, labels)

        for i in range(len(nodes)):
            time = times[i]
            node_data = nodes[i]
            boundary_condition = boundary_conditions[i]
            edge_data = edge_indexs[i]
            edge_attr = attrs[i]
            label = labels[i]
            rcr = rcrs[i]
            p_type = p_types[i]
            e_type = e_types[i]
            edge = edges[i]
            coordinate = coordinates[i]
            branch = branchs[i]
            mydata = MyData(x=torch.tensor(node_data, dtype=torch.float32),
                            boundary_condition=torch.tensor(boundary_condition, dtype=torch.float32),
                            edge_index=torch.tensor(edge_data, dtype=torch.int64),
                            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                            y=torch.tensor(label, dtype=torch.float32),
                            y_mean=torch.tensor(y_mean, dtype=torch.float32),
                            y_std=torch.tensor(y_std, dtype=torch.float32),
                            time=torch.tensor(time, dtype=torch.float32),
                            rcr=torch.tensor(rcr, dtype=torch.float32),
                            p_type=torch.tensor(p_type, dtype=torch.float32),
                            e_type=torch.tensor(e_type, dtype=torch.float32),
                            edge=torch.tensor(edge, dtype=torch.int64),
                            coordinate=torch.tensor(coordinate, dtype=torch.float32),
                            branch=torch.tensor(branch, dtype=torch.int64),
                            )

            data_list.append(mydata)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_train_test_dataloader(args):
    dataset = MyDataset(args.root_path, args.file_path)

    train_dataset, test_dataset = train_test_split(dataset, test_size=(1 - args.train_ratio),
                                                   train_size=args.train_ratio, random_state=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)  # train_loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)  # test_loader

    return train_loader, test_loader



