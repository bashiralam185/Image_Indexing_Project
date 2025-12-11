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

# Cropping
CROP_TOP = 300
CROP_BOTTOM = None

# Dense-mode water suppression
WATER_LINE_FRACTION = 0.15

# Dense-mode umbrella removal
UMBRELLA_S_MIN = 90
UMBRELLA_V_MIN = 120
UMBRELLA_DILATE_SIZE = 9

# Dense-mode texture threshold
TEXTURE_THRESH = 25

SHOW_DEBUG = True
MAX_DEBUG = 10

# Scene classification
PREVIEW_AREA_MIN = 100
DENSE_BLOB_THRESHOLD = 10


# 1. LOAD IMAGES + BACKGROUND

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


# 2. BASE FOREGROUND MASK

def get_foreground_mask_base(img, background):
    diff = cv2.absdiff(img, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, -5
    )
    return mask, gray


# 3. SCENE CLASSIFICATION
def classify_scene(fg_preview):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_preview)
    big_blobs = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= PREVIEW_AREA_MIN:
            big_blobs += 1
    is_dense = (big_blobs >= DENSE_BLOB_THRESHOLD)
    return is_dense, big_blobs


# 4. CLEAN MASK

def clean_mask(mask, ksize=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.medianBlur(mask, 5)
    return mask


# 5. DENSE-MODE FILTERS

def remove_sand(mask, img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sand_mask = cv2.inRange(hsv, (0, 0, 90), (35, 50, 255))
    sand_mask = cv2.medianBlur(sand_mask, 5)
    return cv2.bitwise_and(mask, cv2.bitwise_not(sand_mask)), sand_mask


def remove_umbrellas(mask, img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, sat_mask = cv2.threshold(hsv[:, :, 1], UMBRELLA_S_MIN, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(hsv[:, :, 2], UMBRELLA_V_MIN, 255, cv2.THRESH_BINARY)
    umb = cv2.bitwise_and(sat_mask, val_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (UMBRELLA_DILATE_SIZE, UMBRELLA_DILATE_SIZE))
    umb = cv2.dilate(umb, k)
    return cv2.bitwise_and(mask, cv2.bitwise_not(umb)), umb


def apply_texture_filter(mask, img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap).astype(np.uint8)
    _, texture_mask = cv2.threshold(lap_abs, TEXTURE_THRESH, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(mask, texture_mask), texture_mask


def suppress_water(mask):
    h = mask.shape[0]
    cut = int(WATER_LINE_FRACTION * h)
    out = mask.copy()
    out[:cut] = 0
    return out, cut


# 6. PERSON AREA MODEL

def expected_person_area(y, img_h, dense=True):
    norm = y / float(img_h)
    if dense:
        if norm < 0.33:
            return 40, 600
        elif norm < 0.66:
            return 80, 3000
        else:
            return 200, 10000
    else:
        if norm < 0.33:
            return 60, 600
        elif norm < 0.66:
            return 120, 2500
        else:
            return 260, 8000


# 7. CONNECTED COMPONENTS + SHAPE FILTERS

def extract_people(mask, dense=True):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    h = mask.shape[0]
    pts = []

    for i in range(1, num):
        x, y, w, h_box, area = stats[i]
        cx, cy = centroids[i]

        min_a, max_a = expected_person_area(cy, h, dense=dense)
        if not (min_a <= area <= max_a):
            continue

        ratio = w / float(h_box + 1e-5)

        if dense:
            if ratio < 0.25 or ratio > 3.0:
                continue
            solidity = area / float(w * h_box + 1e-5)
            if solidity < 0.15:
                continue
        else:
            if ratio < 0.4 or ratio > 2.5:
                continue
            solidity = area / float(w * h_box + 1e-5)
            if solidity < 0.3:
                continue

        pts.append((cx, cy))

    return pts, stats, centroids


# 8. ANNOTATIONS (UPDATED FOR BOXES)

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


# 9. MATCHING ACCURACY, True Positves, False Positives, False Negatives

def matching_stats(pred, true, max_dist=20):
    if len(true) == 0:
        # define TP/FP/FN sensibly in the empty-GT case
        TP = 0
        FP = len(pred)
        FN = 0
        acc = 1.0 if len(pred) == 0 else 0.0
        return acc, TP, FP, FN, []

    used = set()
    matches = []

    for j, (px, py) in enumerate(pred):
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
            matches.append((j, best))  # pred j matched GT best
            used.add(best)

    TP = len(matches)
    FN = len(true) - TP
    FP = len(pred) - TP
    rec = TP / (TP + FN)
    prec = TP / (TP + FP)

    return rec, prec, TP, FP, FN, matches

# 10. MAIN PIPELINE

def main():
    imgs, paths = load_images()
    background = build_background(imgs)
    background_cropped, _ = crop(background)

    df = load_annotations()

    predicted_counts = []
    actual_counts = []
    all_rec = []
    all_prec = []
    all_mse = []
    all_tp = []
    all_fp = []
    all_fn = []

    debug_count = 0

    for img_f32, path in zip(imgs, paths):
        img = img_f32.astype(np.uint8)
        cropped, crop_offset = crop(img)

        fg_base, gray = get_foreground_mask_base(cropped, background_cropped)
        fg_preview = clean_mask(fg_base.copy(), ksize=3)

        is_dense, big_blobs = classify_scene(fg_preview)

        # -------- DENSE MODE --------
        if is_dense:
            mode_str = f"DENSE (blobs={big_blobs})"
            fg = fg_base.copy()
            fg = clean_mask(fg, ksize=5)

            fg, sand = remove_sand(fg, cropped)
            fg, umb = remove_umbrellas(fg, cropped)
            fg, tex = apply_texture_filter(fg, cropped)
            fg, waterline = suppress_water(fg)

            pred_pts, stats, cent = extract_people(fg, dense=True)

        # -------- SPARSE MODE --------
        else:
            mode_str = f"SPARSE (blobs={big_blobs})"
            diff = cv2.absdiff(cropped, background_cropped)
            gray_s = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            fg = cv2.adaptiveThreshold(
                gray_s, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                71, 10
            )
            fg = clean_mask(fg, ksize=7)
            fg, waterline = suppress_water(fg)

            pred_pts, stats, cent = extract_people(fg, dense=False)

        true_pts = get_gt_points(df, path, crop_offset, img.shape[0])

        pred_count = len(pred_pts)
        true_count = len(true_pts)

        predicted_counts.append(pred_count)
        actual_counts.append(true_count)

        mse = (pred_count - true_count) ** 2
        rec, prec, TP, FP, FN, matches = matching_stats(
            pred_pts,
            true_pts,
            max_dist=20  # or whatever radius you settled on
        )

        all_mse.append(mse)
        all_rec.append(rec)
        all_prec.append(prec)
        all_tp.append(TP)
        all_fp.append(FP)
        all_fn.append(FN)

        # DEBUG VISUALIZATION
        if SHOW_DEBUG and debug_count < MAX_DEBUG:
            debug_count += 1

            vis = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            for (cx, cy) in pred_pts:
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            for (tx, ty) in true_pts:
                cv2.circle(vis, (int(tx), int(ty)), 5, (255, 0, 0), -1)

            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.title(f"{mode_str}\nMask (Pred {pred_count} / GT {true_count})")
            plt.imshow(fg, cmap="gray")
            plt.axhline(waterline, color='red')
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title(f"{mode_str}\nOverlay (Pred {pred_count} / GT {true_count})")
            plt.imshow(vis)
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    # PLOTS

    x = np.arange(len(predicted_counts))

    # Line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, actual_counts, marker='o', label="Actual")
    plt.plot(x, predicted_counts, marker='x', label="Predicted")
    plt.title("Predicted vs Actual Counts")
    plt.xlabel("Image Index")
    plt.ylabel("People Count")
    plt.grid(True)
    plt.legend()
    plt.savefig("pred_vs_gt.png")
    plt.show()

    # Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_counts, predicted_counts)
    maxv = max(max(actual_counts), max(predicted_counts)) + 5
    plt.plot([0, maxv], [0, maxv], 'r--')
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title("Predicted vs Actual (Scatter)")
    plt.grid(True)
    plt.savefig("pred_vs_gt_scatter.png")
    plt.show()

    # Bar
    plt.figure(figsize=(12, 6))
    width = 0.35
    plt.bar(x - width/2, actual_counts, width)
    plt.bar(x + width/2, predicted_counts, width)
    plt.title("Per-image Predicted vs Actual")
    plt.xlabel("Image Index")
    plt.ylabel("Count")
    plt.xticks(x)
    plt.grid(axis='y')
    plt.savefig("pred_vs_gt_barplot.png")
    plt.show()

    print("\n========= FINAL RESULTS (DUAL MODE) =========")
    print(f"Mean Recall: {np.mean(all_rec) * 100:.2f}%")
    print(f"Mean Precision: {np.mean(all_prec) * 100:.2f}%")
    print(f"Mean MSE: {np.mean(all_mse):.2f}")
    print(f"Mean TP per image: {np.mean(all_tp):.2f}")
    print(f"Mean FP per image: {np.mean(all_fp):.2f}")
    print(f"Mean FN per image: {np.mean(all_fn):.2f}")
    print("=============================================\n")


if __name__ == "__main__":
    main()
