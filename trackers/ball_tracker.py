from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class BallTracker:
    
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path) 
        self.conf_threshold = conf_threshold

    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            # print("Processing frames", i, "to", i + len(batch_frames)-1)
            batch_detections = self.model.predict(batch_frames, self.conf_threshold)
            detections += batch_detections
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = [] 

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # get class names
            cls_names_inv = {v:k for k,v in cls_names.items()} # invert class names dict

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            # Find the ball with highest confidence in this frame
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_names_inv['Ball']:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        print("tracks length:", len(tracks))
        save_stub(stub_path, tracks)
        
        return tracks

    # Remove wrong ball detections based on distance criteria
    def remove_wrong_detections(self, ball_positions):
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        print("Length of ball positions before removing wrong detections:", len(ball_positions))

        # Iterate through ball positions
        for i in range(len(ball_positions)):
            current_bbox = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_bbox) == 0:
                continue
            
            # If this is the first good detection, just record it
            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', None)
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # calculate the distance between the last good bbox and the current position
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i
                
        return ball_positions
    
    # Interpolate missing ball positions to create smooth trajectories
    def interpolate_ball_positions(self, ball_positions):

        print("Length of ball positions before interpolation:", len(ball_positions))

        # Extract ball positions
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions




