import os
import json
import argparse
import torch
import time  

from scripts.image_ranking import ImageRanker
from scripts.garment_detection import GarmentDetector
from scripts.segmentation import GarmentSegmenter
from scripts.mask_refinement import MaskRefiner
from scripts.normalization import GarmentNormalizer
from scripts.validator import GarmentValidator

def run_pipeline(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading AI Models... This might take a moment.")
    ranker = ImageRanker()
    detector = GarmentDetector()
    segmenter = GarmentSegmenter()
    
    # Iterate through all product folders in the data directory
    for product_id in os.listdir(input_dir):
        product_path = os.path.join(input_dir, product_id)
        if not os.path.isdir(product_path):
            continue
            
        print(f"\n--- Processing Product: {product_id} ---")
        
        # --- Start the stopwatch! ---
        start_time = time.time()
        
        # 0. Read Existing Metadata
        source_metadata = {}
        metadata_file = os.path.join(product_path, "Product_metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                source_metadata = json.load(f)
            print(f"[*] Found metadata for: {source_metadata.get('title', 'Unknown')}")
            
        # Extract the title as a variable so Python can use it
        product_title = source_metadata.get("title", "clothing garment")
        
        # 1. Image Ranking
        best_image_path = ranker.select_best_image(product_path, product_title=product_title)
        print(f"[1/5] Selected best image: {os.path.basename(best_image_path)}")
        torch.cuda.empty_cache() # Memory management for GTX 1650
        
        # 2. Garment Detection (NEW: Now unpacking both bbox and bullseye)
        bbox, bullseye = detector.detect(best_image_path, product_title=product_title)
        print(f"[2/5] Detected bounding box: {bbox} | Bullseye Point: {bullseye}")
        torch.cuda.empty_cache() # Memory management for GTX 1650
        
        # 3. Segmentation (NEW: Passing the bullseye point to SAM)
        raw_mask = segmenter.segment(best_image_path, bbox, point=bullseye)
        print("[3/5] Generated SAM segmentation mask.")
        
        # 4. Mask Refinement
        clean_mask = MaskRefiner.refine(raw_mask)
        print("[4/5] Refined and cleaned mask artifacts.")
        
        # 5. Normalization
        final_image = GarmentNormalizer.normalize(best_image_path, clean_mask, target_size=512)
        print("[5/5] Normalized garment to 512x512 transparent asset.")
        
        # 6. Storage
        product_out_dir = os.path.join(output_dir, product_id)
        os.makedirs(product_out_dir, exist_ok=True)
        
        img_out_path = os.path.join(product_out_dir, "clean_garment.png")
        final_image.save(img_out_path, format="PNG")
        
        # 7. Validation & Latency Tracker
        is_valid, validation_msg = GarmentValidator.validate(img_out_path)

        if is_valid:
            print(f"   [+] Validator: PASS ({validation_msg})")
        else:
            print(f"   [-] Validator: FAIL ({validation_msg})")

        end_time = time.time()
        latency_seconds = round(end_time - start_time, 2)

        # Merge Original Data with Pipeline Data AND Validation Status
        final_metadata = {
            **source_metadata, 
            "pipeline_stats": {
                "source_image": os.path.basename(best_image_path),
                "bounding_box_detected": bbox,
                "bullseye_point": bullseye,          # <-- NEW tracking data
                "normalized_size": [512, 512],
                "processing_time_seconds": latency_seconds, 
                "asset_path": img_out_path,
                "is_valid": is_valid,                
                "validation_reason": validation_msg  
            }
        }

        json_out_path = os.path.join(product_out_dir, "garment_metadata.json")
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(final_metadata, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved assets for {product_id} in {latency_seconds} seconds!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Lab Phase-1 Pipeline")
    parser.add_argument("--input", type=str, default="./data/products", help="Path to input products directory")
    parser.add_argument("--output", type=str, default="./outputs", help="Path to outputs directory")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output)