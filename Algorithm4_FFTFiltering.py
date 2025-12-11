import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paths
IMAGES_DIR = 'Data'
ANNOTATIONS_CSV = "annotations.csv"

# Debug display info
SHOW_DEBUG = True
MAX_DEBUG = 4

# Annotation CSV columns
FILENAME_COL = "image_name"
X_COL = "bbox_x"
Y_COL = "bbox_y"
W_COL = "bbox_width"
H_COL = "bbox_height"

def load_images():
    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img.astype(np.float32))
    return imgs, paths

def load_annotations():
    return pd.read_csv(ANNOTATIONS_CSV)



def fft_notch_filter(img_bgr,
                     center_radius=20,
                     peak_threshold_factor=20.0,
                     notch_radius=20,
                     show_plots=True):
    """
    Apply automatic notch filtering in the frequency domain to remove strong
    high-frequency peaks (e.g. repetitive textures).

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (as read by cv2.imread).
    center_radius : int
        Radius around the center (low frequencies) that is always kept.
    peak_threshold_factor : float
        Peaks are defined as frequencies where magnitude > mean + factor * std.
    notch_radius : int
        Radius of each notch around detected peaks (in frequency domain).
    show_plots : bool
        If True, shows original image, spectrum, mask, and filtered result.

    Returns
    -------
    filtered_img_uint8 : np.ndarray
        Filtered grayscale image as uint8.
    """

    # --- 1. Convert to grayscale ---
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr.copy()

    img_gray = img_gray.astype(np.float32)

    # --- 2. FFT and shift zero-frequency to center ---
    F = np.fft.fft2(img_gray)
    Fshift = np.fft.fftshift(F)

    # Magnitude spectrum (for visualization)
    magnitude = np.abs(Fshift)
    magnitude_log = np.log(magnitude + 1)  # log for better visibility

    # --- 3. Build an initial mask that keeps everything ---
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones_like(img_gray, dtype=np.float32)

    # Always keep low frequencies inside center_radius
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    low_freq_region = dist_from_center <= center_radius

    # --- 4. Detect strong high-frequency peaks (outside the low-frequency center) ---
    # Compute statistics only on high-freq region
    high_freq_magnitude = magnitude[~low_freq_region]
    mean_val = high_freq_magnitude.mean()
    std_val = high_freq_magnitude.std()

    # Define threshold for peaks
    peak_threshold = mean_val + peak_threshold_factor * std_val

    # Boolean mask of peaks (high magnitude and outside center)
    peaks_mask = (magnitude > peak_threshold) & (~low_freq_region)

    # --- 5. For every peak, create a notch (zero small neighborhood in mask) ---
    # We'll zero a circular area with radius `notch_radius` around each peak
    peak_positions = np.column_stack(np.where(peaks_mask))

    for (py, px) in peak_positions:
        # Create a circular notch around (py, px)
        y_grid, x_grid = np.ogrid[:rows, :cols]
        dist = np.sqrt((y_grid - py) ** 2 + (x_grid - px) ** 2)
        notch_region = dist <= notch_radius
        mask[notch_region] = 0.0

    # Make sure we **keep** the center low-frequency region
    mask[low_freq_region] = 1.0

    # --- 6. Apply mask in frequency domain ---
    Fshift_filtered = Fshift * mask

    # --- 7. Inverse FFT to return to spatial domain ---
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    img_filtered = np.fft.ifft2(F_ishift)
    img_filtered = np.abs(img_filtered)

    # Normalize to uint8
    img_filtered_norm = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
    filtered_img_uint8 = img_filtered_norm.astype(np.uint8)

    # --- 8. Optional plots ---
    if show_plots:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.title("Original (grayscale)")
        plt.imshow(img_gray, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("Magnitude Spectrum (log)")
        plt.imshow(magnitude_log, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Frequency Mask (1 = keep, 0 = notch)")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("Filtered Image (spatial domain)")
        plt.imshow(filtered_img_uint8, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return filtered_img_uint8

def detect_edges_canny(img_bgr,
                       low_thresh=100,
                       high_thresh=200,
                       blur_ksize=5,
                       show=False):
    """
    Run Canny edge detection on a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image (as from cv2.imread, BGR format).
    low_thresh : int
        Lower hysteresis threshold for Canny.
    high_thresh : int
        Upper hysteresis threshold for Canny.
    blur_ksize : int
        Kernel size for optional Gaussian blur (0 or None to disable).
    show : bool
        If True, show the edge image with matplotlib.

    Returns
    -------
    edges : np.ndarray
        Binary edge image (uint8, 0 or 255).
    """
    # Convert to grayscale
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Optional blur to reduce noise
    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Canny
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    if show:
        plt.figure(figsize=(6, 6))
        plt.title("Canny edges")
        plt.imshow(edges, cmap="gray")
        plt.axis("off")
        plt.show()

    return edges


def connected_components_from_edges(edges,
                                    min_area=0,
                                    show=False):
    """
    Run connected components on a binary edge image.

    Parameters
    ----------
    edges : np.ndarray
        Binary edge image (uint8, typically 0/255 from Canny).
    min_area : int
        Minimum area (in pixels) to keep a component (0 = keep all).
    show : bool
        If True, draw bounding boxes of kept components on a debug image.

    Returns
    -------
    num_labels : int
        Number of labels found (including background label 0).
    labels : np.ndarray
        Label image of the same size as edges (int32).
    stats : np.ndarray
        Per-label statistics: [x, y, w, h, area] for each label.
    centroids : np.ndarray
        (x, y) centroid for each label.
    """
    # Ensure binary (0/1) or 0/255
    # connectedComponentsWithStats expects non-zero = foreground
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Optionally filter out tiny components and visualize
    if show:
        # Make a 3-channel debug image from edges
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for i in range(1, num_labels):  # skip label 0 (background)
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        plt.figure(figsize=(6, 6))
        plt.title("Connected components (boxes)")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    # You can filter by min_area outside this function if you prefer
    return num_labels, labels, stats, centroids

def detect_blobs(img_bgr,
                 min_area=100,
                 max_area=5000,
                 min_threshold=10,
                 max_threshold=200,
                 show=False):
    """
    Detect blobs in a BGR image using SimpleBlobDetector.
    """
    # Convert to grayscale
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold

    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByColor = False

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray)

    if show:
        img_with_blobs = cv2.drawKeypoints(
            gray, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        plt.figure(figsize=(6, 6))
        plt.title(f"Blobs detected: {len(keypoints)}")
        plt.imshow(img_with_blobs, cmap="gray")
        plt.axis("off")
        plt.show()

    return keypoints


def keypoints_to_mask(keypoints, shape, radius_scale=1.0):
    """
    Convert list of keypoints to a binary mask with filled circles.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for kp in keypoints:
        x, y = kp.pt
        r = int(0.5 * kp.size * radius_scale)
        if r < 1:
            r = 1
        cv2.circle(mask, (int(x), int(y)), r, 255, thickness=-1)

    return mask


def connected_components_from_blob_mask(mask,
                                        min_area=0,
                                        show=False):
    """
    Run connected components on a binary blob mask.
    """
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if show:
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        plt.figure(figsize=(6, 6))
        plt.title("Connected components from blob mask")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return num_labels, labels, stats, centroids

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
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    return recall, precision, TP, FP, FN, matches


def main():
    # Load images and annotations (reuse from your original pipeline)
    imgs, paths = load_images()
    df = load_annotations()

    # For final plots
    predicted_counts = []
    actual_counts = []
    all_rec = []
    all_prec = []
    all_mse = []
    all_tp = []
    all_fp = []
    all_fn = []


    debug_count = 0

    # Vertical crop parameters (same style as your alternative code)
    y1 = 420
    y2 = 1920
    MIN_AREA_CC = 20

    # PROCESS EACH IMAGE
    for img_f32, path in zip(imgs, paths):
        # Ensure uint8 BGR image
        img = img_f32.astype(np.uint8)

        # Crop (we treat y1 as crop_offset for GT conversion)
        cropped = img[y1:y2, :]
        crop_offset = y1

        # Apply FFT notch filter (no internal plots here)
        filtered = fft_notch_filter(
            cropped,
            peak_threshold_factor=15,
            notch_radius=15,
            show_plots=False
        )

        # 1) Detect blobs on the filtered image
        keypoints = detect_blobs(
            filtered,
            min_area=100,
            max_area=5000,
            min_threshold=10,
            max_threshold=200,
            show=False
        )

        # 2) Turn keypoints into a binary mask
        mask = keypoints_to_mask(keypoints, filtered.shape)

        # 3) Connected components on that mask
        num_labels, labels, stats, centroids = connected_components_from_blob_mask(
            mask,
            min_area=MIN_AREA_CC,
            show=False    # we'll do our own overlay to match example style
        )

        # 4) Predicted points = centroids of components with sufficient area
        pred_pts = []
        for i in range(1, num_labels):  # skip background label 0
            x, y, w, h, area = stats[i]
            if area < MIN_AREA_CC:
                continue
            cx, cy = centroids[i]
            pred_pts.append((cx, cy))
        pred_pts = np.array(pred_pts, dtype=np.float32)

        # 5) Ground-truth points in cropped coordinates
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

        # DEBUG VISUALIZATION — match style of the example
        if SHOW_DEBUG and debug_count < MAX_DEBUG:
            debug_count += 1

            # RGB version of filtered version for drawing circles
            vis = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

            # Predicted: green, GT: blue/red (same convention as example)
            for (cx, cy) in pred_pts:
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            for (tx, ty) in true_pts:
                cv2.circle(vis, (int(tx), int(ty)), 5, (255, 0, 0), -1)

            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.title(f"Mask (Pred {pred_count} / GT {true_count})")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title(f"Overlay (Pred {pred_count} / GT {true_count})")
            plt.imshow(vis)
            plt.axis("off")

            plt.show()

    # PLOT 1 — LINE PLOT: Predicted vs Actual per image
    plt.figure(figsize=(10, 6))
    x = np.arange(len(predicted_counts))
    plt.plot(x, actual_counts, marker='o', label="Actual")
    plt.plot(x, predicted_counts, marker='x', label="Predicted")
    plt.title("Predicted vs Actual Counts")
    plt.xlabel("Image Index")
    plt.ylabel("People Count")
    plt.legend()
    plt.grid(True)
    plt.savefig("pred_vs_gt_fft_blobs.png")
    plt.show()

    # PLOT 2 — SCATTER: Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_counts, predicted_counts)
    max_val = max(max(actual_counts), max(predicted_counts)) + 5
    plt.plot([0, max_val], [0, max_val], 'r--', label="Ideal")
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title("Predicted vs Actual (Scatter)")
    plt.legend()
    plt.grid(True)
    plt.savefig("pred_vs_gt_fft_blobs_scatter.png")
    plt.show()

    # PLOT 3 — BAR PLOT: Predicted vs Actual per image
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(predicted_counts))

    plt.bar(x - width / 2, actual_counts, width, label="Actual Count")
    plt.bar(x + width / 2, predicted_counts, width, label="Predicted Count")

    plt.title("Per-image Predicted vs Actual People Count (FFT+Blobs)")
    plt.xlabel("Image Index")
    plt.ylabel("Count")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig("pred_vs_gt_fft_blobs_barplot.png")
    plt.show()

    # FINAL METRICS
    print("\n========= FINAL RESULTS (FFT+Blobs) =========")
    print(f"Mean Recall: {np.mean(all_rec) * 100:.2f}%")
    print(f"Mean Precision: {np.mean(all_prec) * 100:.2f}%")
    print(f"Mean MSE: {np.mean(all_mse):.2f}")
    print(f"Mean TP per image: {np.mean(all_tp):.2f}")
    print(f"Mean FP per image: {np.mean(all_fp):.2f}")
    print(f"Mean FN per image: {np.mean(all_fn):.2f}")
    print("Plots saved:")
    print(" - pred_vs_gt_fft_blobs.png")
    print(" - pred_vs_gt_fft_blobs_scatter.png")
    print(" - pred_vs_gt_fft_blobs_barplot.png")
    print("=============================================\n")


if __name__ == "__main__":
    main()
