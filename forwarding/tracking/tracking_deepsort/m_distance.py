import numpy as np
from .kalman_filter import KalmanFilter

"""
Calculate mahalanobis distance cost matrix
"""
def cal_mdistance(dj, yi, si):

    mdistance = np.linalg.multi_dot((dj - yi).T, si, (dj - yi))

    return mdistance


def mdistance_cost(tracks, detections):
    """
    Parameter : 
    ---------
    tracks : list of list[mean, covariance]
    detections : ndarray of [mean], N*4
    -----------
    return : costmatrix with shape (len(tracks), detections.shape[0])
            where entry(i,j) is the m distance between i-th track
            and j-th detection.
    """

    cost_matrix = np.zeros((len(tracks), detections.shape[0]))
    for i in range(len(tracks)):
        temp = KalmanFilter()
        cost_matrix[i, : ] = temp.gating_distance(tracks[i][0], tracks[i][1], detections, False)
    
    return cost_matrix