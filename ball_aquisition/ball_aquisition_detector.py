import sys

sys.path.append('..')
from utils import measure_distance, get_center_of_bbox

class BallAquisitionDetector:
    def __init__(self):
        self.possession_threshold = 50
        self.min_frames = 13
        self.containment_threshold = 0.7

    def get_key_basketball_player_assignment_points(self, player_box, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_box
        width = x2 - x1
        height = y2 - y1

        output_points = []

        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))     

        output_points += [
            (x1, y1), # top left corner
            (x2, y1), # top right corner
            (x1, y2), # bottom left corner
            (x2, y2),  # bottom right corner
            (x1 + width // 2, y1), # top center
            (x1 + width // 2, y2), # bottom center
            (x1, y1 + height // 2), # middle left
            (x2, y1 + height // 2)  # middle right
        ]


        return output_points
    
    def find_minimum_distance_to_ball(self, ball_center, player_box):
        key_points = self.get_key_basketball_player_assignment_points(player_box, ball_center)

        return min([measure_distance(ball_center, point) for point in key_points])

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        player_area = (px2 - px1) * (py2 - py1)
        ball_area = (bx2 - bx1) * (by2 - by1)

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
            return 0.0  # No intersection

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        containment_ratio = intersection_area / ball_area

        return containment_ratio
    
    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('bbox', [])

            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment >= self.containment_threshold:
                high_containment_players.append((player_id, containment))
            else:
                regular_distance_players.append((player_id, min_distance))
        

        # First prioritize players with high containment
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]
        
        # Second prioritize players based on distance
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            return best_candidate[0]
        
        return -1

    def detect_ball_possession(self, player_tracks, ball_tracks):
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        consecutive_possession_count = {}
        
        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue
                
            ball_bbox = ball_info.get('bbox', [])
            if not ball_bbox:
                continue
                
            ball_center = get_center_of_bbox(ball_bbox)
            
            best_player_id = self.find_best_candidate_for_possession(
                ball_center, 
                player_tracks[frame_num], 
                ball_bbox
            )

            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id, 0) + 1
                consecutive_possession_count = {best_player_id: number_of_consecutive_frames} 

                if consecutive_possession_count[best_player_id] >= self.min_frames:
                    possession_list[frame_num] = best_player_id
            else:
                consecutive_possession_count = {}
    
        return possession_list

            








