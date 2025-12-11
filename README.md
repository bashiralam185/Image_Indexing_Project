# Beach Crowd Counting Using Classical Computer Vision

This repository implements a complete classical computer‑vision pipeline for automatic crowd counting on a beach using a fixed camera.  
The system estimates the number of visible people in each image without relying on deep learning, using only traditional image‑processing techniques.

---

## Overview

We developed **three independent algorithms**, each based on a different fundamental approach in classical computer vision:

### • Algorithm 1 — Background Subtraction + Morphology  
### • Algorithm 2 — Blob Detection (SimpleBlobDetector)  
### • Algorithm 3 — Adaptive Sparse/Dense Scene Classification

All algorithms were validated against manually annotated bounding‑box ground truth.

---

## Repository Structure

```
.
├── Algorithm1_BackgroundSubtraction.py
├── Algorithm2_BlobDetector.py
├── Algorithm3_AdaptiveDensity.py
├── Data/
│   ├── <images>.jpg
│   └── annotations.csv
├── README.md
```

---

## Dataset and Annotation Format

Images were manually annotated using **makesense.ai** using **bounding boxes** (not points).

### Annotation CSV Format

| Column | Description |
|--------|-------------|
| image_name | File name of the image |
| bbox_x | X coordinate of bounding box |
| bbox_y | Y coordinate of bounding box |
| bbox_width | Width of bounding box |
| bbox_height | Height of bounding box |

Each bounding‑box center is treated as a ground‑truth person location.

---

# Algorithms

Below is a description of all three implemented algorithms.

---

## Algorithm 1 — Background Subtraction + Morphology

### Description
This method builds a background model from all images, subtracts the background, and extracts moving/foreground regions.  
Several refinement steps help isolate human silhouettes.

### Pipeline
1. Compute background via mean of all images  
2. Subtract background  
3. Convert to grayscale  
4. Adaptive thresholding  
5. Morphological cleaning (opening/closing/median blur)  
6. Umbrella removal using HSV thresholds  
7. Texture filtering (Laplacian)  
8. Waterline suppression  
9. Connected components + shape filtering  
10. Count remaining blobs as detected people  

### Strengths
- Works well when camera and scene are static  
- Good at isolating moving/foreground objects  

### Limitations
- Sensitive to illumination changes  
- Struggles with shadows and umbrellas  
- Not ideal for dense scenes  

---

## Algorithm 2 — Blob Detection (SimpleBlobDetector)

### Description
Detects circular / elliptical regions corresponding to human heads.  
Uses OpenCV's built‑in SimpleBlobDetector.

### Pipeline
1. Convert image to grayscale  
2. Run blob detector with tuned parameters  
3. Convert keypoints to binary mask  
4. Connected components  
5. Count detected blob regions  

### Strengths
- Simple and computationally efficient  
- Works well in sparse scenes  

### Limitations
- Fails in dense crowds where heads overlap  
- More false positives due to sand patterns  

---

## Algorithm 3 — Adaptive Sparse/Dense Scene Classification

### Description
This is the most advanced approach.  
The algorithm classifies the scene as **Sparse** or **Dense**, then applies two different detection pipelines.

### Dense Mode Features
- Relaxed morphological filtering  
- Sand removal  
- Umbrella removal  
- Texture filtering  
- Waterline suppression  
- Larger tolerance on aspect ratio and area  

### Sparse Mode Features
- Stricter thresholding  
- Stronger morphological cleaning  
- Tight constraints on shape and area  

### Final Step
- Connected components  
- Geometric filtering  
- Count final person detections  

### Strengths
- Most robust algorithm  
- Performs well even with occlusion, umbrellas, shadows  

### Limitations
- Higher computational cost  
- Requires careful parameter tuning  

---

# Evaluation Metrics

For every image, we compute:
- **Accuracy = TP / GT**  
- **MSE (Mean Squared Error)**  
- **Predicted vs Actual Count Visualization**

A match is counted when prediction–ground truth distance ≤ 20 pixels.

---

# Running the Algorithms

Each algorithm can be executed individually.

### Algorithm 1
```
python Algorithm1_BackgroundSubtraction.py
```

### Algorithm 2
```
python Algorithm2_BlobDetector.py
```

### Algorithm 3
```
python Algorithm3_AdaptiveDensity.py
```

---

# Requirements

```
opencv-python
numpy
matplotlib
pandas
```
