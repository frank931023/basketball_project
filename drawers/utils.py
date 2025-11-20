import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

# Draws a filled triangle on the given frame at the specified bounding box location
def draw_triangle(frame, bbox, color):
    y= int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

# Draws an ellipse around the given bounding box on the frame.
def draw_ellipse(frame, bbox, color, track_id=None):
    # print("DEBUG bbox =", bbox, type(bbox)) # Debugging line to check bbox value and type
    
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame, 
        center = (x_center, y2), 
        axes = (int(width), int(0.35 * width)), 
        angle=0, 
        startAngle =- 45, 
        endAngle = 235, 
        color = color, 
        thickness = 2, 
        lineType = cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    # print("DEBUG track_id =", track_id) # Debugging line to check track_id value

    if track_id is not None:
        cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)
        
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame
    
