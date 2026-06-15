import cv2
import numpy as np

# --- Global System Memory & Automation Variables ---
prev_left_line = None
prev_right_line = None

# This will dynamically move up and down based on the road terrain and camera angle
dynamic_horizon_y = None

def get_horizon_y(image):
    """Returns the current dynamic horizon, defaulting to 65% of height if uninitialized."""
    global dynamic_horizon_y
    if dynamic_horizon_y is None:
        dynamic_horizon_y = int(image.shape[0] * 0.65)
    return dynamic_horizon_y

def select_region_of_interest(image):
    """Masks the frame adaptively based on the calculated dynamic horizon."""
    height, width = image.shape[:2]
    horizon = get_horizon_y(image)
    
    # A dynamic trapezoid that scales its apex automatically with the horizon line
    polygons = np.array([
        [(int(width * 0.10), height), 
         (int(width * 0.90), height), 
         (int(width * 0.55), horizon), 
         (int(width * 0.45), horizon)]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def make_coordinates(image, line_parameters):
    """Calculates x, y coordinates extending exactly from the bottom to the dynamic horizon."""
    slope, intercept = line_parameters
    height = image.shape[0]
    horizon = get_horizon_y(image)
    
    y1 = height
    y2 = horizon
    
    # Simple line algebra: x = (y - c) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global prev_left_line, prev_right_line, dynamic_horizon_y
    
    left_fit = []
    right_fit = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            # Filter line segments into left/right buckets based on slope angle
            if slope < -0.4 and slope > -2.0: 
                left_fit.append((slope, intercept))
            elif slope > 0.4 and slope < 2.0:
                right_fit.append((slope, intercept))
                
    # Calculate average equations for current frame
    left_avg = np.average(left_fit, axis=0) if left_fit else None
    right_avg = np.average(right_fit, axis=0) if right_fit else None
    
    # --- AUTOMATIC CALIBRATION: Calculate Vanishing Point Intersection ---
    if left_avg is not None and right_avg is not None:
        m1, c1 = left_avg
        m2, c2 = right_avg
        
        # High school algebra: set equations equal to find intersection x and y
        if (m1 - m2) != 0:
            intersect_x = (c2 - c1) / (m1 - m2)
            intersect_y = m1 * intersect_x + c1
            
            # Bound the calculated horizon within a reasonable safe range (50% to 80% of screen)
            min_safe_y = int(image.shape[0] * 0.50)
            max_safe_y = int(image.shape[0] * 0.80)
            intersect_y = max(min_safe_y, min(intersect_y, max_safe_y))
            
            # Smooth the horizon adjustment using an exponential moving average
            current_horizon = get_horizon_y(image)
            dynamic_horizon_y = int(current_horizon * 0.9 + intersect_y * 0.1)

    # Convert equations to physical screen coordinates with historical smoothing
    if left_avg is not None:
        left_line = make_coordinates(image, left_avg)
        if prev_left_line is not None:
            left_line = (prev_left_line * 0.85 + left_line * 0.15).astype(int)
        prev_left_line = left_line
    else:
        left_line = prev_left_line
        
    if right_avg is not None:
        right_line = make_coordinates(image, right_avg)
        if prev_right_line is not None:
            right_line = (prev_right_line * 0.85 + right_line * 0.15).astype(int)
        prev_right_line = right_line
    else:
        right_line = prev_right_line
    
    return [left_line, right_line]

def draw_filled_lane(image, averaged_lines):
    lane_image = np.zeros_like(image)
    if averaged_lines is not None:
        left_line, right_line = averaged_lines
        
        if left_line is not None and right_line is not None:
            # We add a 25-pixel safety margin down from the true intersection 
            # to guarantee the lines can never cross and create an "X"
            safety_offset = 25
            y_top = get_horizon_y(image) + safety_offset
            
            # Recalculate top x coordinates to match the safety offset height
            def adjust_x_to_y(line, target_y):
                if (line[3] - line[1]) == 0: return line[2]
                slope = (line[3] - line[1]) / (line[2] - line[0])
                intercept = line[1] - slope * line[0]
                return int((target_y - intercept) / slope)

            left_x2_adjusted = adjust_x_to_y(left_line, y_top)
            right_x2_adjusted = adjust_x_to_y(right_line, y_top)

            pts = np.array([
                [left_line[0], left_line[1]],       # Bottom left
                [left_x2_adjusted, y_top],          # Top left (capped)
                [right_x2_adjusted, y_top],         # Top right (capped)
                [right_line[0], right_line[1]]      # Bottom right
            ], np.int32)
            
            cv2.fillPoly(lane_image, [pts], (0, 255, 0))
            cv2.line(lane_image, (left_line[0], left_line[1]), (left_x2_adjusted, y_top), (0, 255, 255), 6)
            cv2.line(lane_image, (right_line[0], right_line[1]), (right_x2_adjusted, y_top), (0, 255, 255), 6)

    return lane_image

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120) # Slightly widened sensitivity thresholds
    cropped_edges = select_region_of_interest(edges)
    
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
    
    averaged_lines = average_slope_intercept(frame, lines)
    lane_image = draw_filled_lane(frame, averaged_lines)
    
    final_output = cv2.addWeighted(frame, 1, lane_image, 0.3, 0)
    return final_output

def main():
    # SETUP FOR VIDEO FILE OR LIVE WEBCAM:
    # Set to 0 to use your laptop's integrated webcam: video_path = 0
    video_path = "data/test_video.mp4" 
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        processed_frame = detect_lanes(frame)
        cv2.imshow("Adaptive Autonomous Perception - Lane Tracking", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()