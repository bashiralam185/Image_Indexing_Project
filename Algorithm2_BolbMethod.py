import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def main():
    # Load image (BGR)
    img = cv2.imread("data/1.jpg")
    if img is None:
        raise FileNotFoundError("Could not load data/0.jpg")

    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img_rgb.shape)
    # ---- Vertical crop ----
    # Example: keep only rows y1:y2
    y1 = 420      # top row to keep
    y2 = 1920      # bottom row to keep
    img_crop = img_rgb[y1:y2, :]
    filtered = img_crop
    # 1) Detect blobs
    keypoints = detect_blobs(
        filtered,
        min_area=100,
        max_area=5000,
        show=False  # set True if you also want the blob circles view
    )
    print("Blobs detected:", len(keypoints))

    # 2) Turn keypoints into a binary mask
    mask = keypoints_to_mask(keypoints, filtered.shape)

    # 3) Connected components on that mask
    MIN_AREA_CC = 20
    num_labels, labels, stats, centroids = connected_components_from_blob_mask(
        mask, min_area=MIN_AREA_CC, show=False  # we will do our own overlay
    )

    # 4) Overlay components on top of the original image
    overlay = filtered.copy()

    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]
        if area < MIN_AREA_CC:
            continue
        # draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # draw centroid as a small circle
        cx, cy = centroids[i]
        cv2.circle(overlay, (int(cx), int(cy)), 3, (0, 255, 0), -1)

    # 5) Show original + overlay
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Blobs / components overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    num_components = (stats[1:, 4] >= MIN_AREA_CC).sum()
    print("Connected components (area >= MIN_AREA_CC):", num_components)


if __name__ == "__main__":
    main()
