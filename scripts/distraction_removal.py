import cv2
import numpy as np
from ultralytics import YOLO

class DistractionRemover:
    def __init__(self, model_path='yolov8n-pose.pt'):
        # Automatically downloads the YOLOv8 pose model on first run
        self.model = YOLO(model_path)

    def remove_body_parts(self, image_path, garment_mask, target_product_title):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        
        # Detect human pose landmarks
        results = self.model(image_path, verbose=False)
        
        # If no person is detected, return the original SAM mask
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.xy[0]) == 0:
            return garment_mask 
            
        # Ensure mask is uint8 for OpenCV operations
        garment_mask_uint8 = garment_mask.astype(np.uint8)
        
        # --- NEW: CORE COLOR DETECTION ---
        # 1. Convert image to HSV for robust color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 2. Define standard bounds for human skin in HSV color space
        # These bounds are generally effective across various skin tones
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 3. Create a full-image skin mask
        full_skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
        
        # Extract the 17 keypoints (x, y) for the detected person
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Helper function to get pixel coordinates securely
        def get_pt(idx):
            x, y = keypoints[idx]
            if x > 0 and y > 0 and x < w and y < h:
                return int(x), int(y)
            return None
            
        # COCO Keypoint Mapping:
        # 5: L-Shoulder, 6: R-Shoulder | 9: L-Wrist, 10: R-Wrist
        # 11: L-Hip, 12: R-Hip         | 15: L-Ankle, 16: R-Ankle
            
        # --- 1. ALWAYS REMOVE HEAD & NECK (Generic Box is Safer Here) ---
        l_shoulder, r_shoulder = get_pt(5), get_pt(6)
        if l_shoulder and r_shoulder:
            neck_bottom_y = min(l_shoulder[1], r_shoulder[1]) - int(h * 0.03)
            # Use a box from the top down to just above the shoulders
            head_neck_mask = np.zeros_like(garment_mask_uint8)
            cv2.rectangle(head_neck_mask, (0, 0), (w, max(0, neck_bottom_y)), 255, -1)
            garment_mask_uint8[head_neck_mask == 255] = 0
        
        # --- 2. SURGICAL HAND REMOVAL (Guided by Pose) ---
        # Instead of erasing a generic circle, define a small region *over* the hand
        # and subtract only the skin color within that region.
        
        hand_locations = [get_pt(9), get_pt(10)] # Wrists
        precise_erasing_mask = np.zeros_like(garment_mask_uint8)

        # Scale ROI size by image width, e.g., 10% width
        roi_size = int(w * 0.1)

        for hand_pt in hand_locations:
            if hand_pt:
                x_coord, y_coord = hand_pt
                
                # Create a Region of Interest (ROI) bounding box around the hand
                x_start = max(0, x_coord - roi_size)
                x_end = min(w, x_coord + roi_size)
                y_start = max(0, y_coord - roi_size)
                y_end = min(h, y_coord + roi_size)
                
                # Extract only the relevant skin within that small region
                hand_skin_mask_section = full_skin_mask[y_start:y_end, x_start:x_end]
                
                # Copy that section back into the full Precise Erasing Mask
                precise_erasing_mask[y_start:y_end, x_start:x_end] = hand_skin_mask_section

        # Subtract the precise hand-skin mask from the garment mask
        garment_mask_uint8[precise_erasing_mask == 255] = 0

        # --- DYNAMIC: PRODUCT-TYPE REMOVAL ---
        product_lower = target_product_title.lower()

        # --- 3. DYNAMIC: PANTS/JEANS (Feet Removal) ---
        if "jean" in product_lower or "pant" in product_lower:
            l_ankle, r_ankle = get_pt(15), get_pt(16)
            y_coords = [pt[1] for pt in [l_ankle, r_ankle] if pt]
            if y_coords:
                # Generic rectangle removal from slightly above highest ankle
                lowest_ankle_y = min(y_coords) - int(h * 0.02)
                shoes_mask = np.zeros_like(garment_mask_uint8)
                cv2.rectangle(shoes_mask, (0, max(0, lowest_ankle_y)), (w, h), 255, -1)
                garment_mask_uint8[shoes_mask == 255] = 0
            
        # --- 4. DYNAMIC: SHIRT/TOP (Legs/Pants Removal) ---
        elif "shirt" in product_lower or "top" in product_lower:
            l_hip, r_hip = get_pt(11), get_pt(12)
            y_coords = [pt[1] for pt in [l_hip, r_hip] if pt]
            if y_coords:
                # Generic rectangle removal from slightly below hip
                lowest_hip_y = max(y_coords) + int(h * 0.05)
                pants_mask = np.zeros_like(garment_mask_uint8)
                cv2.rectangle(pants_mask, (0, max(0, lowest_hip_y)), (w, h), 255, -1)
                garment_mask_uint8[pants_mask == 255] = 0

        # Convert back to boolean before returning to the main pipeline
        return garment_mask_uint8.astype(bool)