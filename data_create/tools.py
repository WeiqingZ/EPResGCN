import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def create_edges(points, edges):
    edge = np.zeros((points.shape[0], 1), dtype=int)
    distance = np.zeros((points.shape[0], 1))
    edge[0] = -1
    edge[edges[:, 1].astype(int), 0] = edges[:, 0].astype(int)  # Edge maintains the starting point of the endpoint of a line segment, with index as the endpoint and value as the starting point

    # A branch with the starting point of another branch as the branching point, and the previous node with its branching point as the starting point.
    branch_begin = []
    branch_number = np.max(points[:, 0]).astype(int)
    for i in range(branch_number + 1):  # Get the starting point of all branches
        branch_index = np.where(points[:, 0] == i)[0]
        branch_begin.append(branch_index[0])
    for i in branch_begin:
        if edge[i] in branch_begin:  # If the point before the starting point of some branches is also the starting point, let it be the point before the starting point
            index = branch_begin.index(edge[i])
            edge[i] = edge[branch_begin[index]]

    for i in range(1, edge.shape[0]):
        distance[i] = np.linalg.norm(points[i, 1:4] - points[edge[i], 1:4])

    return edge, distance


def get_ptype_rcr(edge, points, rcr):
    all_point_index = np.arange(len(edge))
    ptype = np.zeros((len(edge), 4))

    inlet = np.array([0])
    ptype[inlet, 0] = 1  # inlet

    unique_elements, counts = np.unique(edge, return_counts=True)
    bifurcation = unique_elements[counts > 1]
    ptype[bifurcation, 2] = 1  # Bifurcation point

    rcrs = np.zeros((len(edge), 2))

    for i in range(rcr.shape[0]):
        outlet = np.where(points[:, 0] == rcr[i, 0])[0][-1]
        rcrs[outlet] = rcr[i, 1:3]
        ptype[outlet, 1] = 1  # outlet

    inner = np.logical_and(ptype[:, 0] != 1, np.logical_and(ptype[:, 1] != 1, ptype[:, 2] != 1))
    ptype[inner, 3] = 1  # Internal point

    return ptype, rcrs


def calculate_three_t(points, dis):
    dis[0] = 0
    branch_len = points.shape[0]
    t = np.zeros((branch_len, 1))

    for i in range(branch_len):
        t[i] = np.sum(dis[:i+1]) / np.sum(dis)

    return t


def resample_points(points, edge, distance, capacity):
    """
    : param points: properties of points
    : param edge: The index of the edge is the endpoint, and the value is the starting point
    : param distance: the length of the edge of index value
    : param capacity: How many points should be left in the parameter caption
    Return: The resampled points, edges, distances, and distances are still true distances, rather than absolute distances between two points after sampling
    """

    inlet = np.array([0])
    number_branch = np.max(points[:, 0]).astype(int)
    for branch_id in range(number_branch + 1):
        branch_index = np.where(points[:, 0] == branch_id)[0]
        while len(branch_index) > capacity:
            all_point_index = np.arange(len(edge))
            outlet = np.setdiff1d(all_point_index, edge)  # New exit location
            unique_elements, counts = np.unique(edge, return_counts=True)
            bifurcation = unique_elements[counts > 1]  # New branch node
            min_index = 0
            min_value = float('inf')
            for i in branch_index:
                if i not in outlet and i not in inlet and i not in bifurcation and distance[i] < min_value:
                    min_index = i
                    min_value = distance[i]

            point_to_replace = edge[min_index]   # The starting point of the shortest edge
            delete_begin = np.where(edge == min_index)[0]  # View edges starting from the endpoint of the shortest edge

            distance[delete_begin] = distance[delete_begin] + distance[min_index]  # The edge starting from the endpoint of the shortest edge plus the length of the shortest edge
            edge[delete_begin] = point_to_replace  # Set a new starting point

            # Delete this point from points
            points = np.delete(points, min_index, 0)
            edge = np.delete(edge, min_index, 0)
            distance = np.delete(distance, min_index, 0)

            index = np.where(edge > min_index)[0]
            edge[index] = edge[index] - 1
            branch_index = np.where(points[:, 0] == branch_id)[0]

    return points, edge, distance
