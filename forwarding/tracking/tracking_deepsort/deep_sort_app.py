
from __future__ import absolute_import
from collections import namedtuple
import numpy as np
import pycocotools.mask as cocomask
from . import preprocessing
from . import nn_matching
from .detection import Detection
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .m_distance import mdistance_cost

TrackElement = namedtuple("TrackElement", ["box", "track_id", "class_", "mask", "score"])

# TODO：options for each class
def track_single_sequence_deepsort(tracker_options, boxes, scores, reids, classes, masks, optical_flow=None):
  # Perform tracking per class and in the end combine the results
  classes_flat = [c for cs in classes for c in cs]
  unique_classes = np.unique(classes_flat) 
  start_track_id = 1
  class_tracks = []
  tracker_options_class = {}

  # tracking per class
  for class_ in unique_classes:
    if class_ == 1:
      tracker_options_class["detection_confidence_threshold"] = tracker_options[
          "detection_confidence_threshold_car"]
      tracker_options_class["lambda"] = tracker_options["lambda_car"]
      tracker_options_class["max_iou_distance"] = tracker_options["max_iou_distance_car"]
      tracker_options_class["gating"] = 0
      cosine_dist = tracker_options["association_threshold_car"] 

    elif class_ == 2:
      tracker_options_class["detection_confidence_threshold"] = tracker_options[
        "detection_confidence_threshold_pedestrian"]
      tracker_options_class["lambda"] = tracker_options["lambda_pedestrain"]
      tracker_options_class["max_iou_distance"] = tracker_options["max_iou_distance_pedestrain"]
      tracker_options_class["gating"] = 1
      cosine_dist = tracker_options["association_threshold_pedestrian"]

    else:
      assert False, "unknown class"
    tracks = run_deepsort(tracker_options_class, boxes, scores, reids, classes, masks, class_, start_track_id,
                                 cosine_dist, optical_flow=optical_flow)
    class_tracks.append(tracks)
    # update the start_track_id
    track_ids_flat = [track.track_id for tracks_t in tracks for track in tracks_t]
    track_ids_flat.append(start_track_id)
    start_track_id = max(track_ids_flat) + 1

  n_timesteps = len(boxes)
  tracks_combined = [[] for _ in range(n_timesteps)]
  for tracks_c in class_tracks:
    for t, tracks_c_t in enumerate(tracks_c):
      tracks_combined[t].extend(tracks_c_t)
  return tracks_combined

def run_deepsort(tracker_options, boxes, scores, reids, classes, masks, class_to_track, start_track_id, 
                 max_cosine_distance, nms_max_overlap=1.0, nn_budget=None, optical_flow=None ):
    # Initialize metric for matching
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # To compute the cost matrix combined cosine distance and m distance
    # because cosine distance need every feature of object, so here cosine means
    # only part of the cost, m distance will directly calculated in the distance metric  
    tracker = Tracker(metric, start_track_id, tracker_options)
    all_tracks = [] 
    all_ids = []
    if optical_flow is None:
        optical_flow = [None for _ in masks]
    else:
        optical_flow = [None] + optical_flow
        assert len(boxes) == len(scores) == len(reids) == len(classes) == len(masks) == len(optical_flow)
    # do tracking frame by frame, n is frame number
    for n, (boxes_n, scores_n, reids_n, classes_n, masks_n, flow_tm1_n) \
        in enumerate(zip(boxes, scores, reids, classes, masks, optical_flow)):
        # make a detection list for current frame
        detections_n = []
        tracks_n = []
        ids_n = []
        for box, score, reid, class_, mask in zip(boxes_n, scores_n, reids_n, classes_n, masks_n):
            box= _to_xywh(box)
            if class_ != class_to_track:
                continue
            if masks is not None and cocomask.area(mask) <= 10:
                continue
            if score >= tracker_options["detection_confidence_threshold"]: 
                detections_n.append(Detection(box, score, reid, class_, mask))  # add feature
            else:
                continue

        boxes_nn = np.array([x.tlwh for x in detections_n])
        scores_nn = np.array([x.confidence for x in detections_n])
        indices = preprocessing.non_max_suppression(
            boxes_nn, nms_max_overlap, scores_nn)
        detections = [detections_n[i] for i in indices]

        # Update tracker
        tracker.predict()
        tracker.update(detections, tracker_options)

        # Convert the result in KITTI format
        for track in tracker.tracks:
            if track.is_deleted() or track.time_since_update > 1:
                continue
            box_tlwh = track.to_tlwh()
            box = _to_xyxy(box_tlwh)
            tracks_n.append(TrackElement(box=box, track_id=track.track_id, mask=track.mask, class_=track.class_, score=track.score))
            ids_n.append(track.track_id)

        all_tracks.append(tracks_n)
        all_ids.append(ids_n)

    all_tracks_new = []
    for i in range(len(all_ids)-1):
        track_t = []
        for j in  range(len(all_ids[i])):
            if not isaobject(all_ids[i][j], all_ids[i+1]): #+all_ids[i+2]
                continue
            track_t.append(all_tracks[i][j])
        all_tracks_new.append(track_t)
    all_tracks_new.append(all_tracks[-1])
                    
    results = [[TrackElement(box=track.box, track_id=track.track_id, mask=track.mask, class_=track.class_,
                             score=track.score) for track in tracks_t] for tracks_t in all_tracks_new]
    
    return results

def _to_xywh(xyxy):
    xywh = xyxy.copy()
    xywh[2:] -= xywh[:2]
    return xywh

def _to_xyxy(xywh):
    xyxy = xywh.copy()
    xyxy[2:] += xywh[:2]
    return xyxy

def isaobject(t_i,t1):	
		if t_i in t1:
			return True
		else:
			return False



class Tracker:
    """
    this class combine all components of deepsort, call maching cascade and iou matching 
    to compute cost metric, and call kalman_filter update and predict the tracks, each track
    will do update and predict.
    """
    def __init__(self, metric, start_track_id, tracker_options, max_age=30, n_init=1):
        self.metric = metric
        self.max_iou_distance = tracker_options["max_iou_distance"]
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = start_track_id

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # predict for each track (mean, covariance)
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, tracker_options):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching process
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, tracker_options)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        # Update the distance metric with new data.
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    # to find matched, unmatched
    def _match(self, detections, tracker_options):

        def gated_metric(tracks, dets, tracker_options, track_indices, detection_indices):
            # cost function, compute distance between track and detection before km
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # cosine distance, here should add mahalanobis distance
            cost_matrix_c = self.metric.distance(features, targets)
            # extract the corresponding mean and covariance of the tracks and dets as list
            track_state = []
            for i in track_indices:
                track_state_n = [tracks[i].mean, tracks[i].covariance]
                track_state.append(track_state_n)
                
            measurement = [dets[i].to_xyah() for i in detection_indices]
            det_mean = np.array(measurement)
            cost_matrix_m = mdistance_cost(track_state, det_mean)
            lbd = tracker_options["lambda"]
            assert cost_matrix_c.shape == cost_matrix_m.shape
            cost_matrix = lbd * cost_matrix_m + (1-lbd) * cost_matrix_c
            
            if tracker_options["gating"] == 1:
                cost_matrix = linear_assignment.gate_cost_matrix(
                              self.kf, cost_matrix, tracks, dets, track_indices,
                              detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # 3 lists of indices
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, tracker_options, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, tracker_options, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_ = detection.class_
        mask = detection.mask
        score = detection.confidence
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            class_, mask, score, detection.feature))
        self._next_id += 1