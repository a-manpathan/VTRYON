import cv2
import numpy as np

class GarmentValidator:
    @staticmethod
    def validate(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if img is None or img.shape[2] != 4:
            return False, "Invalid image format."
        
        alpha = img[:, :, 3]
        h, w = alpha.shape
        total_pixels = h * w
        
        # --- STRICT RULE 1: Fill Ratio ---
        garment_pixels = cv2.countNonZero(alpha)
        fill_ratio = garment_pixels / total_pixels
        
        # Raised from 5% to 12%. Slivers and single legs will now fail.
        if fill_ratio < 0.12:
            return False, f"Empty or tiny extraction (Fill ratio: {fill_ratio:.2f})"
        if fill_ratio > 0.95:
             return False, f"Background removal failed (Fill ratio: {fill_ratio:.2f})"

        # --- NEW STRICT RULE 2: The Center Check ---
        # Look at the middle 30% of the canvas. If there is no fabric there, it's a false positive.
        center_region = alpha[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        if cv2.countNonZero(center_region) < (center_region.size * 0.10):
            return False, "Garment completely missed the center of the image."

        # --- STRICT RULE 3: Fragmentation ---
        _, binary_alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Only count islands that are a decent size (ignores 1-pixel dust)
        large_islands = [c for c in contours if cv2.contourArea(c) > (total_pixels * 0.01)]
        if len(large_islands) > 3:
            return False, f"Garment fragmented into {len(large_islands)} floating pieces."

        return True, "Passed all quality checks."