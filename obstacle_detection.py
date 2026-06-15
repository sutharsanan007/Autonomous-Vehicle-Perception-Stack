import cv2
from ultralytics import YOLO

def main():
    # Load the pre-trained YOLOv8 Nano model (it will download automatically the first time)
    print("Loading AI Model...")
    model = YOLO('yolov8n.pt') 
    
    video_path = "data/test_video.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Feed the frame into the neural network
        # stream=True keeps it fast and memory-efficient
        results = model(frame, stream=True)
        
        # 2. Extract the bounding boxes and draw them
        for r in results:
            # YOLO's built-in plotting function automatically draws boxes and labels!
            annotated_frame = r.plot() 
            
        # Display the result
        cv2.imshow("Autonomous Perception - Obstacle Detection", annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()