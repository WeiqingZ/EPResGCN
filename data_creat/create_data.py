import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from tools import *


save_path = "D:\\Grom\\river_data\\data_5_inrcr_29\\"
folder_path = "D:\\Grom\\river_data\\edgesandpoints_29"
title = ['time'] + ['x'] + ['y'] + ['z'] + ['A'] + ['inlet'] + ['outlet'] + ['bif'] + ['inner'] + ['r1r2'] + \
        ['c'] + ['edge'] + ['distance'] + ['u'] + ['p'] + ['branch_id']


for dir in os.listdir(folder_path):
    dirs = os.path.join(folder_path, dir)
    edges = pd.read_csv(os.path.join(dirs, "edges_pro.csv")).values  # edges
    for file_dir in os.listdir(dirs):
        file_path = os.path.join(dirs, file_dir)
        if not os.path.isdir(file_path):
            continue
        rcr = pd.read_csv(os.path.join(file_path, "rcr.csv")).values
        index = pd.read_csv(os.path.join(file_path, "index.csv")).values
        edge, distance = create_edges(index, edges)
        cap = 10
        points, edge, distance = resample_points(index, edge, distance, cap)
        ptype, rcr = get_ptype_rcr(edge, points, rcr)
        data = []
        t = 0.005
        dt = 0.005
        for i in range(1901, 2001):
            num = str(i).zfill(4)
            file_name = "1D_" + num + ".csv"
            one_d_data = pd.read_csv(os.path.join(file_path, file_name)).values
            branch_num = np.max(one_d_data[:, 0:1])
            data_feature = np.zeros((points.shape[0], 3))

            for branch_id in range(branch_num + 1):
                branch_index_1d = np.where(one_d_data[:, 0:1] == branch_id)[0]
                one_d_t = np.linspace(0, 1, len(branch_index_1d))
                branch_section = interp1d(one_d_t, one_d_data[branch_index_1d, 2], kind='cubic')
                branch_u = interp1d(one_d_t, one_d_data[branch_index_1d, 3], kind='cubic')
                branch_p = interp1d(one_d_t, one_d_data[branch_index_1d, 4], kind='cubic')

                branch_index_3d = np.where(points[:, 0:1] == branch_id)[0]
                branch_3d = points[branch_index_3d, 1:4]
                dis_3d = distance[branch_index_3d, 0:1]
                three_d_t = calculate_three_t(branch_3d, dis_3d)
                data_feature[branch_index_3d, 0:1] = branch_section(three_d_t)
                data_feature[branch_index_3d, 1:2] = branch_u(three_d_t)
                data_feature[branch_index_3d, 2:3] = branch_p(three_d_t)

            time = np.full((points.shape[0], 1), t)
            data_t = np.concatenate((time, points[:, 1:4], data_feature[:, 0:1], ptype, rcr,
                                     edge, distance, data_feature[:, 1:3], points[:, 0:1]), axis=1)
            t = t + dt
            data.append(data_t)

        data = np.concatenate(data, axis=0)
        df = pd.DataFrame(data)
        df = pd.concat([pd.DataFrame([title], columns=df.columns), df], ignore_index=True)
        df.to_csv(save_path + dir + '_' + file_dir + '.csv', index=False, header=False)
