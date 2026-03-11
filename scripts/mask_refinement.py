import cv2
import numpy as np

class MaskRefiner:
    @staticmethod
    def refine(binary_mask):
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        
        # The Steamroller (Bridges pattern gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Extract Outer Silhouette
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask_uint8
            
        # Fill the garment solidly
        largest_contour = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(closed_mask)
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        return filled_mask