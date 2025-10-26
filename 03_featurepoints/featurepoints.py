# ==========================================================
# Nama: Angelica Kierra Ninta Gunting
# NIM: 13522048
# Fitur: Feature Points (Harris, SIFT, FAST)
# ==========================================================

import cv2
import numpy as np
import pandas as pd
from skimage import data, color, io, img_as_ubyte, transform
import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)

# === Get timestamp ===
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

# === Utility: save Resulrs ===
def save_compare(title, original, processed, label_text="Original | Processed"):
    h, w = original.shape[:2]
    combined = np.hstack((original, processed))
    if len(combined.shape) == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    border_h = 40
    white_border = np.ones((border_h, combined.shape[1], 3), dtype=np.uint8) * 255
    final_img = np.vstack((combined, white_border))
    cv2.putText(final_img, label_text, (10, h + border_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    timestamp = get_timestamp()
    filename = f"outputs/output_{timestamp}_{title}.png"
    cv2.imwrite(filename, img_as_ubyte(final_img))
    print(f"Saved: {filename}")

# === Feature Detection Methods ===
def apply_harris(img, block_size=2, ksize=3, k=0.04, thresh_ratio=0.01):
    gray = np.float32(img)
    harris = cv2.cornerHarris(gray, block_size, ksize, k)
    img_harris = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = harris > thresh_ratio * harris.max()
    img_harris[mask] = [0, 0, 255]
    num_corners = np.sum(mask)
    mean_response = float(np.mean(harris[mask])) if num_corners > 0 else 0.0
    total_response = float(np.sum(harris[mask])) if num_corners > 0 else 0.0
    return img_harris, int(num_corners), mean_response, total_response

def apply_sift(img, contrastThreshold=0.04, edgeThreshold=10, nOctaveLayers=3, sigma=1.6):
    sift = cv2.SIFT_create(contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold,
                           nOctaveLayers=nOctaveLayers,
                           sigma=sigma)
    kp, des = sift.detectAndCompute(img, None)
    img_sift = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    num_kp = len(kp)
    if num_kp > 0:
        mean_response = np.mean([k.response for k in kp])
        mean_size = np.mean([k.size for k in kp])
        mean_angle = np.mean([k.angle for k in kp])
        total_response = np.sum([k.response for k in kp])
    else:
        mean_response, mean_size, mean_angle, total_response = 0, 0, 0, 0
    return img_sift, num_kp, mean_response, mean_size, mean_angle, total_response

def apply_fast(img, threshold=25, nonmaxSuppression=True):
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmaxSuppression)
    kp = fast.detect(img, None)
    img_fast = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
    num_kp = len(kp)
    if num_kp > 0:
        mean_response = np.mean([k.response for k in kp])
        mean_size = np.mean([k.size for k in kp])
        mean_angle = np.mean([k.angle for k in kp])
        total_response = np.sum([k.response for k in kp])
    else:
        mean_response, mean_size, mean_angle, total_response = 0, 0, 0, 0
    return img_fast, num_kp, mean_response, mean_size, mean_angle, total_response

# === Helper ===
def load_datasets():
    return {
        "camera": data.camera(),
        "coins": data.coins(),
        "checkerboard": data.checkerboard(),
        "astronaut": color.rgb2gray(data.astronaut()),
        "chelsea": color.rgb2gray(data.chelsea())
    }

def load_personal():
    personal_dir = "personal"
    if not os.path.exists(personal_dir):
        print("No 'personal/' folder found.")
        return {}

    imgs = {}
    for filename in os.listdir(personal_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(personal_dir, filename)
            img = io.imread(path)
            if img.ndim == 3:
                img = color.rgb2gray(img)

            max_size = 300
            h, w = img.shape[:2]
            scale = max_size / max(h, w)
            if scale < 1:
                img = transform.rescale(img, scale, anti_aliasing=True)

            imgs[os.path.splitext(filename)[0]] = img_as_ubyte(img)
    return imgs

# === Run Experiment ===
def run_experiment(dataset_type, method_type):
    stats = []
    timestamp = get_timestamp()

    if dataset_type == "preset":
        all_images = load_datasets()
        print("\nAvailable preset datasets:")
        for i, key in enumerate(all_images.keys(), start=1):
            print(f"{i}. {key}")
        print(f"{len(all_images)+1}. all (use all)")
        choice = input("Choose which image(s) to use [number/comma/all]: ").strip()
        if choice == str(len(all_images)+1) or choice.lower() == "all":
            selected_images = all_images
        else:
            selected_keys = [list(all_images.keys())[int(i)-1] for i in choice.split(",") if i.strip().isdigit()]
            selected_images = {k: all_images[k] for k in selected_keys if k in all_images}
    else:
        selected_images = load_personal()

    if not selected_images:
        print("No images found. Exiting.")
        return

    if method_type in ["harris", "all"]:
        print("\n--- Harris Parameters ---")
        block_size = int(input("Block size [default 2]: ") or "2")
        ksize = int(input("Sobel kernel size [default 3]: ") or "3")
        k = float(input("Harris k value [default 0.04]: ") or "0.04")
        thresh_ratio = float(input("Threshold ratio [default 0.01]: ") or "0.01")

    if method_type in ["sift", "all"]:
        print("\n--- SIFT Parameters ---")
        contrast = float(input("Contrast threshold [default 0.04]: ") or "0.04")
        edge = float(input("Edge threshold [default 10]: ") or "10")
        octave = int(input("Number of octave layers [default 3]: ") or "3")
        sigma = float(input("Sigma [default 1.6]: ") or "1.6")

    if method_type in ["fast", "all"]:
        print("\n--- FAST Parameters ---")
        threshold = int(input("FAST threshold [default 25]: ") or "25")
        nms_input = input("Use non-max suppression? [y/n, default y]: ").lower() or "y"
        nonmax = True if nms_input == "y" else False

    for name, img in selected_images.items():
        print(f"\nProcessing image: {name}")
        if img.dtype != np.uint8:
            img = img_as_ubyte(img)

        if method_type in ["harris", "all"]:
            img_harris, num, mean_resp, total_resp = apply_harris(img, block_size, ksize, k, thresh_ratio)
            save_compare(f"{timestamp}_{dataset_type}_{name}_harris", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                         img_harris, f"Harris ({num} pts)")
            stats.append(["Harris", name, num, mean_resp, "-", "-", total_resp])

        if method_type in ["sift", "all"]:
            img_sift, num, mean_resp, mean_size, mean_angle, total_resp = apply_sift(img, contrast, edge, octave, sigma)
            save_compare(f"{timestamp}_{dataset_type}_{name}_sift", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                         img_sift, f"SIFT ({num} pts)")
            stats.append(["SIFT", name, num, mean_resp, mean_size, mean_angle, total_resp])

        if method_type in ["fast", "all"]:
            img_fast, num, mean_resp, mean_size, mean_angle, total_resp = apply_fast(img, threshold, nonmax)
            save_compare(f"{timestamp}_{dataset_type}_{name}_fast", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                         img_fast, f"FAST ({num} pts)")
            stats.append(["FAST", name, num, mean_resp, mean_size, mean_angle, total_resp])

    out_file = f"outputs/stats_featurepoints_{dataset_type}_{method_type}_{timestamp}.csv"
    df = pd.DataFrame(stats, columns=["Method", "Image", "Num_Features", "Mean_Response",
                                      "Mean_Size", "Mean_Angle", "Total_Response"])
    df.to_csv(out_file, index=False)
    print(f"\nExperiment complete. Statistics saved to {out_file}\n")

# === Interactive Menu ===
def main():
    print("=== Feature / Corner Detection ===")
    print("1. Preset Dataset (cameraman, coins, etc.)")
    print("2. Personal Photos (in 'personal/' folder)")
    choice = input("Choose dataset type [1/2]: ").strip()
    dataset_type = "preset" if choice == "1" else "personal"

    print("\nSelect method:")
    print("1. Harris Corner Detection")
    print("2. SIFT Feature Detection")
    print("3. FAST Feature Detection")
    print("4. All Methods")
    method_choice = input("Choose method [1/2/3/4]: ").strip()

    method_type = {
        "1": "harris",
        "2": "sift",
        "3": "fast",
        "4": "all"
    }.get(method_choice, None)

    if method_type:
        run_experiment(dataset_type, method_type)
    else:
        print("Invalid choice. Exiting.")

# === Entry point ===
if __name__ == "__main__":
    main()
