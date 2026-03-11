import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ImageRanker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=dtype).to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # THE REFINED PROMPTS: Punishing "cropped" and "close up"
        self.prompts = [
            "A clear, full front view showing the complete clothing garment from top to bottom.", 
            "A heavily cropped close-up, a back view, or a folded garment." 
        ]

    def select_best_image(self, image_dir, product_title="clothing"):
        valid_exts = ('.png', '.jpg', '.jpeg')
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        if len(image_paths) == 1:
            return image_paths[0]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(text=self.prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1) 
            best_idx = probs[:, 0].argmax().item()
            
        return image_paths[best_idx]