# ==========================================================
# Nama: Angelica Kierra Ninta Gunting
# NIM: 13522048
# Fitur: Edge Detection (Sobel & Canny)
# ==========================================================

import cv2
import numpy as np
import pandas as pd
from skimage import data, color, io, img_as_ubyte, transform
import os

os.makedirs("outputs", exist_ok=True)

# === Save Results ===
def save_compare(title, original, processed, label_text="Original | Processed"):
    h, w = original.shape
    combined = np.hstack((original, processed))

    if len(combined.shape) == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    border_h = 40
    white_border = np.ones((border_h, combined.shape[1], 3), dtype=np.uint8) * 255
    final_img = np.vstack((combined, white_border))

    cv2.putText(final_img, label_text, (10, h + border_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    filename = f"outputs/output_{title}.png"
    cv2.imwrite(filename, img_as_ubyte(final_img))
    print(f"Saved: {filename}")

# === Edge Detection Functions ===
def apply_sobel(img, ksize=3):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel / np.max(sobel) * 255).astype(np.uint8)
    return sobel

def apply_canny(img, lower=50, upper=150):
    return cv2.Canny(img, lower, upper)

# === Helper: Load preset dataset ===
def load_datasets():
    return {
        "camera": data.camera(),
        "coins": data.coins(),
        "checkerboard": data.checkerboard(),
        "astronaut": color.rgb2gray(data.astronaut()),
        "chelsea": color.rgb2gray(data.chelsea())
    }

# === Helper: Load and resize personal images ===
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

# === Main experiment runner ===
def run_experiment(dataset_type, method_type):
    params = []

    if dataset_type == "preset":
        all_images = load_datasets()
        print("\nAvailable preset datasets:")
        for i, key in enumerate(all_images.keys(), start=1):
            print(f"{i}. {key}")
        print(f"{len(all_images) + 1}. all (use all preset images)")

        choice = input("Choose which image(s) to use [number or comma-separated]: ").strip()
        if choice == str(len(all_images) + 1) or choice.lower() == "all":
            selected_images = all_images
        else:
            selected_keys = [list(all_images.keys())[int(i.strip()) - 1] for i in choice.split(",") if i.strip().isdigit()]
            selected_images = {k: all_images[k] for k in selected_keys if k in all_images}
    else:
        selected_images = load_personal()

    if not selected_images:
        print("No images found. Exiting.")
        return

    for name, img in selected_images.items():
        print(f"\nProcessing image: {name}")
        if img.dtype != np.uint8:
            img = img_as_ubyte(img)

        # === SOBEL ===
        if method_type in ["sobel", "both"]:
            k_values = [int(k.strip()) for k in (input("Enter Sobel kernel sizes (comma separated, odd numbers): ") or "3,5,7").split(",")]
            for k in k_values:
                sobel = apply_sobel(img, k)
                save_compare(f"{dataset_type}_{name}_sobel_k{k}",
                             img, sobel, f"Sobel Edge (ksize={k})")
                params.append(["Sobel", name, f"k={k}", "-"])

        # === CANNY ===
        if method_type in ["canny", "both"]:
            thresholds_input = input("Enter Canny thresholds as pairs (e.g., 30-100,50-150,100-200): ") or "30-100,50-150,100-200"
            threshold_pairs = []
            for pair in thresholds_input.split(","):
                if "-" in pair:
                    low, high = pair.split("-")
                    threshold_pairs.append((int(low.strip()), int(high.strip())))
            for lower, upper in threshold_pairs:
                canny = apply_canny(img, lower, upper)
                save_compare(f"{dataset_type}_{name}_canny_l{lower}_u{upper}",
                             img, canny, f"Canny Edge (lower={lower}, upper={upper})")
                params.append(["Canny", name, lower, upper])

    out_file = f"outputs/params_edge_{dataset_type}_{method_type}.csv"
    df = pd.DataFrame(params, columns=["Method", "Image", "Param1", "Param2"])
    df.to_csv(out_file, index=False)
    print(f"\nExperiment complete. Parameters saved to {out_file}\n")

# === Interactive menu ===
def main():
    print("=== Edge Detection ===")
    print("1. Preset Dataset (cameraman, coins, etc.)")
    print("2. Personal Photos (in 'personal/' folder)")
    choice = input("Choose dataset type [1/2]: ").strip()
    dataset_type = "preset" if choice == "1" else "personal"

    print("\nSelect method:")
    print("1. Sobel Edge Detection")
    print("2. Canny Edge Detection")
    print("3. Both Sobel and Canny")
    method_choice = input("Choose method [1/2/3]: ").strip()

    method_type = {
        "1": "sobel",
        "2": "canny",
        "3": "both"
    }.get(method_choice, None)

    if method_type:
        run_experiment(dataset_type, method_type)
    else:
        print("Invalid choice. Exiting.")

# === Entry point ===
if __name__ == "__main__":
    main()
