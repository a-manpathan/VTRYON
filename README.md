
# Product Lab: Phase 1 - Automated Garment Extraction

A zero-shot computer vision pipeline designed to automatically isolate, extract, and normalize clothing garments from raw studio photography. This generates clean 512x512 transparent assets optimized for Phase 2 Virtual Try-On (VTO) generation.

## 🧠 Pipeline Architecture
1. **Image Ranking (CLIP):** Automatically selects the best product shot (preferring flat lays over model shots).
2. **Pose-Guided Detection (YOLOv8):** Uses human skeletal tracking to surgically isolate the upper or lower half of the body based on JSON metadata.
3. **Zero-Shot Segmentation (SAM):** Traces the garment using targeted "Bullseye" point prompting.
4. **Mask Refinement (OpenCV):** Bridges intricate fabric patterns and strictly defines the outer silhouette.
5. **Quality Validation:** Mathematically grades the extraction (checking fill ratios and fragmentation) to prevent bad data from reaching the VTO engine.

---

## ⚙️ Installation

**1. Install Core Libraries**
```bash
pip install torch torchvision transformers ultralytics opencv-python pillow numpy

```

**2. Install Segment Anything (SAM)**

```bash
pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)

```

**3. Download SAM Weights**
Download the ViT-B model weights (`sam_vit_b_01ec64.pth`) from the official Meta repository and place it in a `./models/` directory.

---

## 📁 Required Folder Structure

The pipeline expects your input data to be organized in folders per product:

```text
project_root/
├── data/
│   └── products/
│       ├── product_id_01/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── Product_metadata.json
├── models/
│   └── sam_vit_b_01ec64.pth
├── scripts/
│   └── (all .py module files)
└── main.py

```

---

## 🚀 Usage

To run the full extraction and validation pipeline on your dataset, execute `main.py` from the root directory:

```bash
python main.py --input ./data/products --output ./outputs

```

**Outputs:**
For each processed product, the script generates a new folder in `./outputs` containing:

* `clean_garment.png`: The normalized 512x512 transparent asset.
* `garment_metadata.json`: The merged original metadata, latency stats, bounding box coordinates, and the Validator's Pass/Fail flag.

```
