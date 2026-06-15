import cv2
import numpy as np

def select_region_of_interest(image):
    height, width = image.shape[:2]
    # Narrowed the bottom right coordinate slightly to avoid guardrails
    polygons = np.array([
        [(int(width * 0.15), height), (int(width * 0.85), height), (int(width * 0.5), int(height * 0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def make_coordinates(image, line_parameters):
    """Calculates the x and y coordinates for a single solid line"""
    slope, intercept = line_parameters
    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.6) # Matches the height of our ROI
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """Combines all broken line segments into one solid left and right line"""
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: continue # Prevent division by zero
        
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Filter by slope to separate left and right, and ignore horizontal noise
        if slope < -0.5: 
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))
            
    # Average them out and create coordinates
    left_line = make_coordinates(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.average(right_fit, axis=0)) if right_fit else None
    
    return [left_line, right_line]

def draw_filled_lane(image, averaged_lines):
    """Draws a green polygon filling the space between the two lanes"""
    lane_image = np.zeros_like(image)
    
    if averaged_lines is not None:
        left_line, right_line = averaged_lines
        
        # Ensure both lines were detected before drawing
        if left_line is not None and right_line is not None:
            # Create a 4-point polygon connecting the left and right lines
            pts = np.array([
                [left_line[0], left_line[1]],   # Bottom left
                [left_line[2], left_line[3]],   # Top left
                [right_line[2], right_line[3]], # Top right
                [right_line[0], right_line[1]]  # Bottom right
            ], np.int32)
            
            # Fill the space with green
            cv2.fillPoly(lane_image, [pts], (0, 255, 0))
            
            # Optional: Draw bright yellow borders on the actual lines
            cv2.line(lane_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 8)
            cv2.line(lane_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 255), 8)

    return lane_image

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = select_region_of_interest(edges)
    
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    
    # Run our new math functions
    averaged_lines = average_slope_intercept(frame, lines)
    lane_image = draw_filled_lane(frame, averaged_lines)
    
    # Blend the green area (30% opacity) with the original frame
    final_output = cv2.addWeighted(frame, 1, lane_image, 0.3, 0)
    
    return final_output

def main():
    video_path = "data/test_video.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        processed_frame = detect_lanes(frame)
        cv2.imshow("Autonomous Perception - Lane Tracking", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()