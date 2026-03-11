import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class GarmentSegmenter:
    def __init__(self, model_path='./models/sam_vit_b_01ec64.pth', model_type='vit_b'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    # NEW: Accept the point coordinate from the detector
    def segment(self, image_path, bbox, point=None):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image_rgb)
        input_box = np.array(bbox)
        
        # If we have a bullseye, format it for SAM
        if point is not None:
            point_coords = np.array([point])
            point_labels = np.array([1]) # "1" tells SAM this point is the foreground object
        else:
            point_coords = None
            point_labels = None
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks[0]