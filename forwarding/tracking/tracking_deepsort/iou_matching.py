from __future__ import absolute_import
import numpy as np
from . import linear_assignment
import pycocotools.mask as cocomask
from cv2 import remap, INTER_NEAREST

def iou_cost(tracks, detections, flow_tm1_t, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix

def iou_cost_mask(tracks, detections, flow_tm1_t, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    h, w = flow_tm1_t.shape[:2]
    flow_tm1_t = -flow_tm1_t
    flow_tm1_t[:, :, 0] += np.arange(w)
    flow_tm1_t[:, :, 1] += np.arange(h)[:, np.newaxis]

    masks_t = []
    masks_tm1 = []
    masks_tm1_warped = [warp_flow(mask, flow_tm1_t) for mask in masks_tm1]

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        mask_tm1 = tracks[track_idx].mask
        candidate = np.asarray([detections[i].mask for i in detection_indices])
        masks_t.append(candidate)
        mask_tm1.append(mask_tm1)
    masks_tm1_warped = [warp_flow(mask, flow_tm1_t) for mask in masks_tm1]
    mask_ious = cocomask.iou(masks_t, masks_tm1_warped, [False] * len(masks_tm1_warped))
    cost_matrix += mask_ious
    return cost_matrix

def warp_flow(mask_as_rle, flow):
  # unpack
  mask = cocomask.decode([mask_as_rle])
  # warp
  warped = _warp(mask, flow)
  # pack
  packed = cocomask.encode(np.asfortranarray(warped))
  return packed

def _warp(img, flow):
  # for some reason the result is all zeros with INTER_LINEAR...
  # res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
  res = remap(img, flow, None, INTER_NEAREST)
  res = np.equal(res, 1).astype(np.uint8)
  return res

def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)