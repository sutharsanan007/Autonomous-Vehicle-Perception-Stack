import cv2
import numpy as np

# --- NEW: Global variables to give our system "memory" ---
prev_left_line = None
prev_right_line = None

def select_region_of_interest(image):
    height, width = image.shape[:2]
    # FIX: Pushed the horizon down to 75% of the screen height
    polygons = np.array([
        [(int(width * 0.15), height), (int(width * 0.85), height), (int(width * 0.5), int(height * 0.75))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    height = image.shape[0]
    y1 = height
    # FIX: Must match the 75% horizon from the mask above!
    y2 = int(height * 0.75) 
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global prev_left_line, prev_right_line
    
    left_fit = []
    right_fit = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue
            
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            if slope < -0.5: 
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
                
    # FIX 2: Add Memory and Smoothing (Exponential Moving Average)
    if left_fit:
        left_line = make_coordinates(image, np.average(left_fit, axis=0))
        if prev_left_line is not None:
            # Blend 80% old line with 20% new line for buttery smooth tracking
            left_line = (prev_left_line * 0.8 + left_line * 0.2).astype(int)
        prev_left_line = left_line
    else:
        left_line = prev_left_line # Fallback to memory if blinded
        
    if right_fit:
        right_line = make_coordinates(image, np.average(right_fit, axis=0))
        if prev_right_line is not None:
            right_line = (prev_right_line * 0.8 + right_line * 0.2).astype(int)
        prev_right_line = right_line
    else:
        right_line = prev_right_line # Fallback to memory if blinded
    
    return [left_line, right_line]

def draw_filled_lane(image, averaged_lines):
    lane_image = np.zeros_like(image)
    if averaged_lines is not None:
        left_line, right_line = averaged_lines
        
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],   
                [left_line[2], left_line[3]],   
                [right_line[2], right_line[3]], 
                [right_line[0], right_line[1]]  
            ], np.int32)
            
            cv2.fillPoly(lane_image, [pts], (0, 255, 0))
            cv2.line(lane_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 8)
            cv2.line(lane_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 255), 8)

    return lane_image

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = select_region_of_interest(edges)
    
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    
    averaged_lines = average_slope_intercept(frame, lines)
    lane_image = draw_filled_lane(frame, averaged_lines)
    
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