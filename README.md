# DeepSORT based on TrackR-CNN
Code for DeepSORT based on detector from TrackR-CNN for the Multi Object Tracking and Segmentation (MOTS) task.  
Tracking part author: Zhiye Wen, Yan Wang


## Paper
### TrackR-CNN
https://www.vision.rwth-aachen.de/media/papers/mots-multi-object-tracking-and-segmentation/MOTS.pdf

### DeepSORT
https://arxiv.org/abs/1703.07402

## Running this code

### Folder structure and config flags
Our codes are stored in the `forwarding/tracking/tracking_deepsort`, the main function `deep_sort_app` is imported to `forwarding/tracking/TrackingForwarder.py` to combine with other parts in Track R-CNN. The results of detection are stored in `forwarded/conv3d_sep2/detection/5/` as the input of tracking.

### Run tracking

You can use the following command to run the our tracking algorithm and to obtain final results in the `forwarded/conv3d_sep2/tracking_data` :
```
python main.py configs/conv3d_sep2 "{\"build_networks\":false,\"import_detections\":true,\"task\":\"forward_tracking\",\"dataset\":\"KITTI_segtrack_feed\",\"do_tracking\":true,\"visualize_detections\":false,\"visualize_tracks\":false,\"load_epoch_no\":5,\"video_tags_to_load\":[\"0002\",\"0006\",\"0007\",\"0008\",\"0010\",\"0013\",\"0014\",\"0016\",\"0018\"]}"
```
You can also visualize the tracking results here by setting `visualize_tracks` to true, and results will be stored in `forwarded/conv3d_sep2/vis/`.

### Evaluation
Run the script for the evaluation on the validation set

To evaluate, run
```
python mots_eval/eval.py forwarded/conv3d_sep2/tracking_data gt_folder val.seqmap
```
where "val.seqmap" is a textfile containing the sequences which you want to evaluate on. 

### Tuning
The script for random tuning will find the best combination of tracking parameters on the training set and then evaluate these parameters on the validation set.

To use this script, run
```
python segtrack_tune_experiment.py forwarded/conv3d_sep2/detections/5 /srv/store/dlenv/home/users/pp5-y7s/Tr4_mahal/gt/instances_txt / /srv/store/dlenv/home/users/pp5-y7s/Tr4_mahal/evalresult /srv/store/dlenv/home/users/pp5-y7s/Tr4_mahal/tmp_folder mots_eval/ reid num_iterations
```
where `/forwarded/conv3d_sep2/detections/5/` is a folder containing the model output on the training set (obtained by the forwarding command above); `/mots_eval//` refers to the official evaluation script; `reid` is association_type; `num_iterations` is the number of random trials (1000 in the paper Track R-CNN); `/gt/instances_txt/ ` refers to the `instances` or `instances_txt` folder containing the annotations (which you can download from the project website); at `/evalresult`, a file will be created containing the results of the individual tuning iterations, please make sure this path is writable; at `/tmp_folder` a lot of intermediate folders will be stored.

## References
Parts of this code are based on Nwojke(https://github.com/nwojke/deep_sort/tree/master/deep_sort) 
