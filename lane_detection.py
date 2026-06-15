import cv2
import numpy as np

def select_region_of_interest(image):
    height, width = image.shape[:2]
    # Triangle mask focused on the immediate driving lane
    polygons = np.array([
        [(int(width * 0.1), height), (int(width * 0.9), height), (int(width * 0.5), int(height * 0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, lines):
    """
    Creates a transparent black layer and draws the calculated mathematical lines 
    onto it in solid bright green.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw a green line with a thickness of 5 pixels
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def detect_lanes(frame):
    # 1. Classical preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = select_region_of_interest(edges)
    
    # 2. Hough Line Transform: Turn white edge pixels into distinct lines
    # Adjust these parameters if lines are too flickering or missing
    lines = cv2.HoughLinesP(
        cropped_edges, 
        rho=2, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=40, 
        maxLineGap=5
    )
    
    # 3. Create a green overlay image of the detected lines
    line_image = draw_lines(frame, lines)
    
    # 4. Blend the green lines onto our original color driving video frame
    final_output = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return final_output

def main():
    video_path = "data/test_video.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = detect_lanes(frame)
        
        # Display the beautiful combined result
        cv2.imshow("Autonomous Perception - Lane Tracking", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()