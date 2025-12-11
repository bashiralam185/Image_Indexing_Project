import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIGURATION

IMAGES_DIR = "Data"
ANNOTATIONS_CSV = "annotations.csv"

# Annotation CSV columns
FILENAME_COL = "image_name"
X_COL = "bbox_x"
Y_COL = "bbox_y"
W_COL = "bbox_width"
H_COL = "bbox_height"

# Image cropping
CROP_TOP = 300
CROP_BOTTOM = None

# Water suppression
WATER_LINE_FRACTION = 0.20

# Umbrella suppression
UMBRELLA_S_MIN = 70
UMBRELLA_V_MIN = 80
UMBRELLA_DILATE_SIZE = 9

# Texture threshold
TEXTURE_THRESH = 5

SHOW_DEBUG = True
MAX_DEBUG = 10


# 1. LOAD IMAGES + BACKGROUND MODEL

def load_images():
    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img.astype(np.float32))
    return imgs, paths


def build_background(imgs):
    return np.mean(imgs, axis=0).astype(np.uint8)


def crop(img):
    h = img.shape[0]
    y1 = CROP_TOP
    y2 = CROP_BOTTOM if CROP_BOTTOM is not None else h
    return img[y1:y2], y1


# 2. FOREGROUND MASK

def get_foreground_mask(img, background):
    diff = cv2.absdiff(img, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, -10
    )
    return mask


# 3. CLEAN MASK

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)
    return mask


# 4. UMBRELLA REMOVAL

def remove_umbrellas(mask, img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, sat_mask = cv2.threshold(hsv[:,:,1], UMBRELLA_S_MIN, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(hsv[:,:,2], UMBRELLA_V_MIN, 255, cv2.THRESH_BINARY)

    umb = cv2.bitwise_and(sat_mask, val_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (UMBRELLA_DILATE_SIZE, UMBRELLA_DILATE_SIZE))
    umb = cv2.dilate(umb, kernel)

    umb_inv = cv2.bitwise_not(umb)
    return cv2.bitwise_and(mask, umb_inv), umb


# 5. TEXTURE FILTER

def apply_texture_filter(mask, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.uint8)

    _, texture_mask = cv2.threshold(lap_abs, TEXTURE_THRESH, 255, cv2.THRESH_BINARY)
    refined = cv2.bitwise_and(mask, texture_mask)
    return refined, texture_mask


# 6. WATER SUPPRESSION

def suppress_water(mask):
    h = mask.shape[0]
    limit = int(WATER_LINE_FRACTION * h)
    out = mask.copy()
    out[:limit] = 0
    return out, limit


# 7. CONNECTED COMPONENTS + FILTERING

def expected_person_area(y, img_h):
    norm = y / float(img_h)
    if norm < 0.33:
        return 25, 400
    elif norm < 0.66:
        return 60, 1800
    else:
        return 150, 8000


def extract_people(mask):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    h = mask.shape[0]
    pts = []

    for i in range(1, num):
        x, y, w, h_box, area = stats[i]
        cx, cy = centroids[i]

        # -------- 1) Loosened area thresholds --------
        min_a, max_a = expected_person_area(cy, h)
        min_a *= 0.5     # allow smaller
        max_a *= 1.5     # allow larger

        if not (min_a <= area <= max_a):
            continue

        # -------- 2) Relaxed shape ratio filter --------
        ratio = w / float(h_box + 1e-5)
        if ratio < 0.15 or ratio > 6.0:
            continue

        # -------- 3) Reduced minimum blob size --------
        if w < 2 or h_box < 2:
            continue

        pts.append((cx, cy))

    return pts, stats, centroids



# 8. LOAD ANNOTATIONS

def load_annotations():
    return pd.read_csv(ANNOTATIONS_CSV)


def get_gt_points(df, filename, crop_offset, img_h):
    base = os.path.basename(filename)
    sub = df[df[FILENAME_COL] == base]

    if sub.empty:
        return np.zeros((0, 2), dtype=np.float32)

    cx = sub[X_COL].values + sub[W_COL].values / 2
    cy = sub[Y_COL].values + sub[H_COL].values / 2

    cy -= crop_offset
    return np.array([(x, y) for x, y in zip(cx, cy) if y >= 0], dtype=np.float32)


# 9. MATCHING ACCURACY

def matching_accuracy(pred, true, max_dist=20):
    if len(true) == 0:
        return 1.0 if len(pred) == 0 else 0.0

    used = set()
    match = 0

    for px, py in pred:
        best = None
        best_d = 99999
        for i, (tx, ty) in enumerate(true):
            if i in used:
                continue
            d = np.hypot(px - tx, py - ty)
            if d < best_d:
                best_d = d
                best = i
        if best_d <= max_dist:
            match += 1
            used.add(best)

    return match / len(true)


# 10. MAIN — FULL PIPELINE + ALL PLOTS

def main():
    imgs, paths = load_images()
    background = build_background(imgs)
    background_cropped, _ = crop(background)

    df = load_annotations()

    # For final plots
    predicted_counts = []
    actual_counts = []
    all_acc = []
    all_mse = []

    debug_count = 0

    # PROCESS EACH IMAGE

    for img_f32, path in zip(imgs, paths):
        img = img_f32.astype(np.uint8)
        cropped, crop_offset = crop(img)

        # Pipeline
        fg = get_foreground_mask(cropped, background_cropped)
        fg = clean_mask(fg)
        fg2, umb = remove_umbrellas(fg, cropped)
        fg3, tex = apply_texture_filter(fg2, cropped)
        fg4, water_line = suppress_water(fg3)

        pred_pts, _, _ = extract_people(fg4)
        true_pts = get_gt_points(df, path, crop_offset, img.shape[0])

        pred_count = len(pred_pts)
        true_count = len(true_pts)

        predicted_counts.append(pred_count)
        actual_counts.append(true_count)

        mse = (pred_count - true_count)**2
        acc = matching_accuracy(pred_pts, true_pts)

        all_mse.append(mse)
        all_acc.append(acc)

        # DEBUG VISUALIZATION
        if SHOW_DEBUG and debug_count < MAX_DEBUG:
            debug_count += 1

            vis = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            for (cx, cy) in pred_pts:
                cv2.circle(vis, (int(cx), int(cy)), 5, (0,255,0), -1)
            for (tx, ty) in true_pts:
                cv2.circle(vis, (int(tx), int(ty)), 5, (255,0,0), -1)

            plt.figure(figsize=(14,6))

            plt.subplot(1,2,1)
            plt.title(f"Mask (Pred {pred_count} / GT {true_count})")
            plt.imshow(fg4, cmap="gray")
            plt.axhline(water_line, color="red")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.title(f"Overlay (Pred {pred_count} / GT {true_count})")
            plt.imshow(vis)
            plt.axis("off")

            plt.show()

    # PLOT 1 — LINE PLOT: Predicted vs Actual per image
    plt.figure(figsize=(10,6))
    x = np.arange(len(predicted_counts))
    plt.plot(x, actual_counts, marker='o', label="Actual")
    plt.plot(x, predicted_counts, marker='x', label="Predicted")
    plt.title("Predicted vs Actual Counts")
    plt.xlabel("Image Index")
    plt.ylabel("People Count")
    plt.legend()
    plt.grid(True)
    plt.savefig("pred_vs_gt.png")
    plt.show()

    # PLOT 2 — SCATTER: Predicted vs Actual

    plt.figure(figsize=(6,6))
    plt.scatter(actual_counts, predicted_counts)
    max_val = max(max(actual_counts), max(predicted_counts)) + 5
    plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal")
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title("Predicted vs Actual (Scatter)")
    plt.legend()
    plt.grid(True)
    plt.savefig("pred_vs_gt_scatter.png")
    plt.show()

    # PLOT 3 — BAR PLOT: Predicted vs Actual per image

    plt.figure(figsize=(12,6))
    width = 0.35
    x = np.arange(len(predicted_counts))

    plt.bar(x - width/2, actual_counts, width, label="Actual Count")
    plt.bar(x + width/2, predicted_counts, width, label="Predicted Count")

    plt.title("Per-image Predicted vs Actual People Count")
    plt.xlabel("Image Index")
    plt.ylabel("Count")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig("pred_vs_gt_barplot.png")
    plt.show()

    # FINAL METRICS

    print("\n========= FINAL RESULTS =========")
    print(f"Mean Accuracy: {np.mean(all_acc)*100:.2f}%")
    print(f"Mean MSE: {np.mean(all_mse):.2f}")
    print("Plots saved:")
    print(" - pred_vs_gt.png")
    print(" - pred_vs_gt_scatter.png")
    print(" - pred_vs_gt_barplot.png")
    print("=================================\n")


if __name__ == "__main__":
    main()
