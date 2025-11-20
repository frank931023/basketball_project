import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance, get_foot_position

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]

    # Validate detected keypoints based on proportions with tactical view keypoints
    def validate_keypoints(self, keypoints_list):
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]
            
            # Get indices of detected keypoints (not (0, 0))
            # Also ensure the index exists in self.key_points to avoid out-of-range accesses
            detected_indices = [i for i, kp in enumerate(frame_keypoints)
                                if kp[0] > 0 and kp[1] > 0 and i < len(self.key_points)]
            
            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            # Validate each detected keypoint
            for i in detected_indices:
                # Skip if this is (0, 0)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Choose two other detected keypoints that are not marked invalid
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Calculate distances between corresponding tactical keypoints
                # Guard against any unexpected index issues just in case
                if i >= len(self.key_points) or j >= len(self.key_points) or k >= len(self.key_points):
                    continue
                # else:
                #     print("Not out of range")

                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                # Calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical
                    error = abs(error)

                    if error >0.8:  # 80% error margin                        
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list


    #　Transform player positions from video frame coordinates to tactical view coordinates
    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            frame_keypoints = frame_keypoints.xy.tolist()[0]

            # Skip frames with insufficient keypoints
            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Get detected keypoints for this frame
            detected_keypoints = frame_keypoints
            
            # Filter out undetected keypoints (those with coordinates (0,0)).
            # Also ensure the index exists in `self.key_points` to avoid out-of-range accesses
            # valid_indices = [i for i, kp in enumerate(detected_keypoints)
            #                  if kp[0] > 0 and kp[1] > 0 and i < len(self.key_points)]
            safe_valid_indices = [i for i in range(len(detected_keypoints)) 
                      if i < len(self.key_points) and detected_keypoints[i][0] > 0 and detected_keypoints[i][1] > 0]

            
            # Need at least 4 points for a reliable homography
            if len(safe_valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Create source and target point arrays for homography
            # (use only indices that are valid for both detected_keypoints and self.key_points)
            try:
                source_points = np.array([detected_keypoints[i] for i in safe_valid_indices], dtype = np.float32)
                target_points = np.array([self.key_points[i] for i in safe_valid_indices], dtype = np.float32)
            except IndexError:
                # In case of unexpected index problems, skip this frame
                tactical_player_positions.append(tactical_positions)
                continue
            
            try:
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                # Transform each player's position
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    # Use bottom center of bounding box as player position
                    player_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position)

                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                        continue

                    tactical_positions[player_id] = tactical_position[0].tolist()
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions
