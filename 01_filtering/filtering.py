# ==========================================================
# Nama: Angelica Kierra Ninta Gunting
# NIM: 13522048
# Fitur: Filtering (Gaussian & Median)
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

# === Filters ===
def apply_gaussian(img, ksize=5, sigma=1.2):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_median(img, ksize=5):
    return cv2.medianBlur(img, ksize)

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

# === Main Experiment Runner ===
def run_experiment(dataset_type, experiment_type):
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

    # === Process each selected image ===
    for name, img in selected_images.items():
        print(f"\nProcessing image: {name}")

        if img.dtype != np.uint8:
            img = img_as_ubyte(img)

        # === Gaussian - vary sigma ===
        if experiment_type == "gaussian_sigma":
            k = int(input("Enter fixed kernel size (odd number, e.g., 5): ") or "5")
            sigmas = [float(s.strip()) for s in (input("Enter sigma values (comma separated): ") or "1.0,5.0,15.0").split(",")]

            for sigma in sigmas:
                filtered = apply_gaussian(img, k, sigma)
                save_compare(f"{dataset_type}_{name}_gaussian_fixK{k}_s{sigma}",
                             img, filtered, f"Gaussian (k={k}, sigma={sigma})")
                params.append(["Gaussian_fixK", name, k, sigma])

        # === Gaussian - vary kernel ===
        elif experiment_type == "gaussian_kernel":
            sigma = float(input("Enter fixed sigma (e.g., 1.2): ") or "1.2")
            kernels = [int(k.strip()) for k in (input("Enter kernel sizes (comma separated, odd number): ") or "3,5,9,15").split(",")]
            for k in kernels:
                filtered = apply_gaussian(img, k, sigma)
                save_compare(f"{dataset_type}_{name}_gaussian_fixS{sigma}_k{k}",
                             img, filtered, f"Gaussian (sigma={sigma}, k={k})")
                params.append(["Gaussian_fixS", name, k, sigma])

        # === Median filter ===
        elif experiment_type == "median":
            kernels = [int(k.strip()) for k in (input("Enter kernel sizes (comma separated, odd number): ") or "3,5,9").split(",")]
            for k in kernels:
                filtered = apply_median(img, k)
                save_compare(f"{dataset_type}_{name}_median_k{k}",
                             img, filtered, f"Median (k={k})")
                params.append(["Median", name, k, "-"])

        else:
            print("Invalid experiment type.")
            return

    out_file = f"outputs/params_filtering_{dataset_type}_{experiment_type}.csv"
    df = pd.DataFrame(params, columns=["Filter", "Image", "Kernel_Size", "Sigma"])
    df.to_csv(out_file, index=False)
    print(f"\nExperiment complete. Parameters saved to {out_file}\n")

# === Interactive Menu ===
def main():
    print("=== Image Filtering ===")
    print("1. Preset Dataset (cameraman, coins, etc.)")
    print("2. Personal Photos (in 'personal/' folder)")
    choice = input("Choose dataset type [1/2]: ")

    dataset_type = "preset" if choice.strip() == "1" else "personal"

    print("\nSelect experiment type:")
    print("1. Gaussian (vary sigma, fixed kernel size)")
    print("2. Gaussian (vary kernel size, fixed sigma)")
    print("3. Median filter (vary kernel size)")
    exp_choice = input("Choose experiment [1/2/3]: ")

    experiment_type = {
        "1": "gaussian_sigma",
        "2": "gaussian_kernel",
        "3": "median"
    }.get(exp_choice, None)

    if experiment_type:
        run_experiment(dataset_type, experiment_type)
    else:
        print("Invalid choice. Exiting.")

# === Entry point ===
if __name__ == "__main__":
    main()
