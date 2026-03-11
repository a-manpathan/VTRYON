import cv2
import numpy as np
from PIL import Image

class GarmentNormalizer:
    @staticmethod
    def normalize(image_path, refined_mask, target_size=512):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create 4-channel image (RGBA)
        b_channel, g_channel, r_channel = cv2.split(img)
        alpha_channel = refined_mask
        rgba = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        
        # Find bounding box of the refined mask to crop tightly
        coords = cv2.findNonZero(refined_mask)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = rgba[y:y+h, x:x+w]
        
        # Pad to square while maintaining aspect ratio
        max_dim = max(w, h)
        pad_y = (max_dim - h) // 2
        pad_x = (max_dim - w) // 2
        
        # Pad with transparent pixels (0,0,0,0)
        padded = cv2.copyMakeBorder(cropped, pad_y, max_dim - h - pad_y, pad_x, max_dim - w - pad_x, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        
        # Resize to standardized target size
        final_asset = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(final_asset)