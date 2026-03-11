import cv2
from ultralytics import YOLO

class GarmentDetector:
    def __init__(self, pose_model='yolov8n-pose.pt'):
        self.model = YOLO(pose_model)

    def detect(self, image_path, product_title=""):
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        fallback_box = [int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)]
        fallback_point = [int(w/2), int(h/2)] 
        
        results = self.model(image_path, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return fallback_box, fallback_point
            
        person_box = results[0].boxes.xyxy[0].cpu().numpy()
        px_min, py_min, px_max, py_max = person_box
        
        if results[0].keypoints is None or len(results[0].keypoints.xy[0]) == 0:
            return [int(px_min), int(py_min), int(px_max), int(py_max)], fallback_point
            
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        def get_pt(idx):
            x, y = keypoints[idx]
            return (x, y) if x > 0 and y > 0 else None

        nose = get_pt(0)
        hip_l, hip_r = get_pt(11), get_pt(12)
        sh_l, sh_r = get_pt(5), get_pt(6)
        
        waist_ys = [pt[1] for pt in [hip_l, hip_r] if pt is not None]
        waist_y = sum(waist_ys) / len(waist_ys) if waist_ys else None
        
        shoulder_ys = [pt[1] for pt in [sh_l, sh_r] if pt is not None]
        shoulder_y = sum(shoulder_ys) / len(shoulder_ys) if shoulder_ys else None
        
        neck_y = nose[1] if nose else (shoulder_y - (h * 0.15) if shoulder_y else 0)
            
        title_lower = product_title.lower()
        is_bottom = any(word in title_lower for word in ["pant", "jean", "trouser", "short", "skirt", "bottom", "cargo"])
        is_top = any(word in title_lower for word in ["shirt", "top", "jacket", "hoodie", "sweater", "t-shirt"])
        
        pad_x = (px_max - px_min) * 0.05
        final_xmin = max(0, px_min - pad_x)
        final_xmax = min(w, px_max + pad_x)
        center_x = (final_xmin + final_xmax) / 2
        
        if is_top and waist_y and shoulder_y:
            final_ymin = max(0, int(neck_y))
            final_ymax = min(h, int(waist_y + (h * 0.08))) 
            
            # THE FIX: Move the pin 30% from the left edge, avoiding the bare chest/V-neck
            safe_x = final_xmin + ((final_xmax - final_xmin) * 0.3)
            bullseye = [int(safe_x), int(shoulder_y + ((waist_y - shoulder_y) * 0.3))]
            
            return [int(final_xmin), final_ymin, int(final_xmax), final_ymax], bullseye
            
        elif is_bottom and waist_y:
            final_ymin = max(0, int(waist_y - (h * 0.15))) 
            final_ymax = int(py_max)
            
            # Bottoms are safe to keep dead center
            bullseye = [int(center_x), int(waist_y + (h * 0.1))]
            
            return [int(final_xmin), final_ymin, int(final_xmax), final_ymax], bullseye
            
        else:
            return [int(px_min), int(py_min), int(px_max), int(py_max)], fallback_point