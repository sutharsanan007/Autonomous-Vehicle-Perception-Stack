import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# Hardware Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing YOLOP AI on hardware: {device.type.upper()}...")

yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
yolop_model = yolop_model.to(device) 
yolop_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_lanes(frame):
    h_orig, w_orig = frame.shape[:2]

    img_resized = cv2.resize(frame, (640, 640))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        _, da_seg_out, ll_seg_out = yolop_model(img_tensor)

    da_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    ll_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    da_mask_resized = cv2.resize(da_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    ll_mask_resized = cv2.resize(ll_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # --- THE EGO-LANE FLOOD FILL ALGORITHM ---
    # 1. Create a copy of the drivable area and use lane lines as "walls" (value 0)
    drivable_walls = da_mask_resized.copy()
    drivable_walls[ll_mask_resized == 1] = 0  

    h, w = drivable_walls.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    seed_point = (int(w / 2), int(h - 30))  # Start "pouring" right in front of the car

    ego_lane_mask = np.zeros_like(da_mask_resized)

    # 2. Pour the paint! If the seed point is road (1), fill the enclosed area with a 2
    if drivable_walls[seed_point[1], seed_point[0]] == 1:
        cv2.floodFill(drivable_walls, ff_mask, seed_point, 2)
        ego_lane_mask[drivable_walls == 2] = 1 # Extract only the painted ego-lane
    else:
        ego_lane_mask = da_mask_resized # Fallback if blocked

    # Paint the final visual overlay
    overlay = np.zeros_like(frame)
    overlay[ego_lane_mask == 1] = (0, 255, 0)    # Paint Ego Lane Green
    overlay[ll_mask_resized == 1] = (0, 0, 255)  # Paint ALL lines Red

    # --- FOOLPROOF DEVIATION MATH ---
    # Look at the bottom 50 pixels of the isolated Ego Lane
    bottom_ego_pixels = np.where(ego_lane_mask[h-50:h, :] == 1)[1]
    
    deviation = 0.0
    is_warning = False
    car_center = w / 2

    if len(bottom_ego_pixels) > 0:
        # Get the strict left and right boundaries of the green carpet
        left_edge = np.min(bottom_ego_pixels)
        right_edge = np.max(bottom_ego_pixels)
        
        lane_center = (left_edge + right_edge) / 2
        deviation = ((lane_center - car_center) / car_center) * 100
        is_warning = abs(deviation) > 15.0

    stats = {
        "deviation": deviation,
        "is_warning": is_warning,
        "road_status": "YOLOP Ego-Lane Active",
        "brightness": int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    }
    
    return overlay, stats