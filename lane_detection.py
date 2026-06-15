import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# 1. Load the pre-trained YOLOP model from PyTorch Hub
print("Loading YOLOP AI for Semantic Road Segmentation...")
yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
yolop_model.eval()  # Set model to evaluation (inference) mode

# 2. Standard image normalization required by YOLOP
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_lanes(frame):
    """Feeds the frame into YOLOP and extracts the drivable area and lane lines."""
    h_orig, w_orig = frame.shape[:2]

    # Preprocess: YOLOP expects a 640x640 image tensor
    img_resized = cv2.resize(frame, (640, 640))
    img_tensor = transform(img_resized).unsqueeze(0)

    # Run AI Inference
    with torch.no_grad():
        # YOLOP outputs: Object boxes, Drivable Area mask, and Lane Line mask
        _, da_seg_out, ll_seg_out = yolop_model(img_tensor)

    # FIX: Use argmax across the channel dimension (dim=1) to collapse 
    # the probability channels down to a flat 2D image [640, 640]
    da_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    ll_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Resize the flat 2D masks back to your video's original resolution
    da_mask_resized = cv2.resize(da_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    ll_mask_resized = cv2.resize(ll_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Create the blank overlay and paint the pixel classes
    overlay = np.zeros_like(frame)
    overlay[da_mask_resized == 1] = (0, 255, 0)   # Paint drivable road green
    overlay[ll_mask_resized == 1] = (0, 0, 255)   # Paint lane lines red

    # --- ADAS DASHBOARD LOGIC ---
    # Look at the bottom section of the lane mask to compute cross-track deviation
    bottom_red_pixels = np.where(ll_mask_resized[h_orig-50:h_orig, :] == 1)[1]
    deviation = 0.0
    is_warning = False
    
    if len(bottom_red_pixels) > 0:
        lane_center = np.mean(bottom_red_pixels)
        car_center = w_orig / 2
        deviation = ((lane_center - car_center) / (w_orig / 2)) * 100
        is_warning = abs(deviation) > 15.0

    stats = {
        "deviation": deviation,
        "is_warning": is_warning,
        "road_status": "YOLOP Semantic AI Active",
        "brightness": int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    }
    
    return overlay, stats