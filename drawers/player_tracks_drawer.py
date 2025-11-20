from .utils import draw_ellipse, draw_triangle

class PlayerTracksDrawer:

    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.default_player_team_id = 1 # Default team ID for players
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    # Draw player tracks on video frames 
    def draw(self, video_frames, tracks, player_assignment, ball_aquisition):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks[frame_num]

            player_assignment_for_frame = player_assignment[frame_num]
            # print(f"Frame {frame_num}: Player assignment -", player_assignment_for_frame)

            player_id_has_ball = ball_aquisition[frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)

                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color

                if track_id == player_id_has_ball:
                    frame = draw_triangle(frame, player['bbox'], (0, 0, 255)) # Red triangle for ball possession
                
                frame = draw_ellipse(frame, player['bbox'], color=color, track_id=track_id)

            output_video_frames.append(frame)

        return output_video_frames
        