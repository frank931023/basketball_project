from utils import read_video, save_video
import argparse, os
from trackers import PlayerTracker, BallTracker
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer,
    CourtKeypointDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer
)
from team_assigner.team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_cal import SpeedAndDistanceCal
from ai_agent_export import AiAgentExporter

from config import (
    STUBS_DEFUALT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEOS_PATH
)


def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    # parser.add_argument('input_video', type=str, default='input_videos/video_3.mp4', help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEOS_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFUALT_PATH,
                        help='Path to stub directory')
    parser.add_argument('--export_agent_json', type=str, default=os.path.join('ai_agent_summary.json'),
                        help='Path to export AI agent per-frame JSON summary')
    return parser.parse_args()


def main():

    args = parse_args()

    # read video
    video_frames = read_video('input_videos/video_1.mp4')

    # Initialize trackers
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH, conf_threshold=0.000000005)

    # Initialize court keypoint detector
    try:
        court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)
        print("Successfully loaded court keypoint detector model.")
    except Exception as e:
        print("Failed to load court keypoint detector model:", e)
        return

    # Get player and ball tracks
    player_tracks = player_tracker.get_object_tracks(
        video_frames, 
        read_from_stub=True, 
        stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
    )
    
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames, 
        read_from_stub=True, 
        stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
    )

    # Get court keypoints
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames, 
        read_from_stub=True, 
        stub_path=os.path.join(args.stub_path, 'court_keypoint_stubs.pkl')
    )

    # print("Court keypoints:", court_keypoints_per_frame)

    # Remove wrong ball detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    # Interpolate ball positions
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Initialize team assigner
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,     
        player_tracks, 
        read_from_stub=True, 
        stub_path=os.path.join(args.stub_path, 'player_team_assignment_stubs.pkl')
    )

    # Ball acquisition detection
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(
        player_tracks, 
        ball_tracks
    )

    # Detrct passes and interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, player_assignment)

    # Tactical view conversion
    tactical_view_converter = TacticalViewConverter('images/basketball_court.png')
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(
        court_keypoints_per_frame, 
        player_tracks
    )

    # Speed and Distance Calculation
    speed_and_distance_calculator = SpeedAndDistanceCal(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )

    player_distance_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distance_per_frame)

    # Initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    # Draw Object tracks
    print("Drawing player tracks...")
    output_video_frames = player_tracks_drawer.draw(
        video_frames, 
        player_tracks, 
        player_assignment, 
        ball_aquisition
    )

    # print("Drawing ball tracks...")
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # Draw team ball control
    output_video_frames = team_ball_control_drawer.draw(
        output_video_frames, 
        player_assignment, 
        ball_aquisition
    )

    # Draw passes and interceptions
    output_video_frames = pass_interception_drawer.draw(
        output_video_frames, 
        passes, 
        interceptions
    )

    # Draw court keypoints
    output_video_frames = court_keypoint_drawer.draw(
        output_video_frames,
        court_keypoints_per_frame
    )

    # Tactical view drawing
    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        player_assignment,
        ball_aquisition
    )

    # Draw speed and distance
    output_video_frames = speed_and_distance_drawer.draw(
        output_video_frames,
        player_tracks,
        player_distance_per_frame,
        player_speed_per_frame
    )

    # save video
    save_video(output_video_frames, args.output_video)

    
    # Build and export per-frame summary for AI agent
    try:
        exporter = AiAgentExporter()
        summary = exporter.build_summary_per_frame(
            frames=range(len(video_frames)),
            timestamps=None,
            tactical_player_positions=tactical_player_positions,
            ball_tracks=ball_tracks,
            ball_aquisition=ball_aquisition,
            player_assignment=player_assignment,
            player_speeds=player_speed_per_frame,
            recent_events=None,
            court_keypoints=court_keypoints_per_frame,
        )
        exporter.export_summary_json(args.export_agent_json, summary)
        print(f"Exported AI agent summary to {args.export_agent_json}")
    except Exception as e:
        print("Failed to export AI agent summary:", e)

    

if __name__ == "__main__":
    main()


