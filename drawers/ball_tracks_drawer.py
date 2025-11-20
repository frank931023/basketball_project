from .utils import draw_triangle

class BallTracksDrawer:

    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)  # Green color for ball pointer

    # Draw ball tracks on video frames
    def draw(self, video_frames, tracks):
        output_video_frames = []
        print("Starting to draw ball tracks...")
        print("Total frames to process for ball tracks:", len(video_frames))
        print("Total tracks available:", len(tracks))

        for frame_num, frame in enumerate(video_frames):
            
            frame = frame.copy()
            ball_dict = tracks[frame_num]
            # print(f"Frame {frame_num}: Ball dict -", ball_dict)

            # Draw Balls
            for _, ball in ball_dict.items():
                if ball["bbox"] is None:
                    print("skip ball drawing")
                    continue
                frame = draw_triangle(frame, ball["bbox"], self.ball_pointer_color)
            
            output_video_frames.append(frame)

        return output_video_frames
        