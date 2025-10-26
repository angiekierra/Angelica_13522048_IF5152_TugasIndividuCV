# ==========================================================
# Nama: Angelica Kierra Ninta Gunting
# NIM: 13522048
# Fitur: Geometric Transformation
# ==========================================================

import cv2
import numpy as np
import pandas as pd
from skimage import data, color, io, img_as_ubyte, transform
from datetime import datetime
import os

os.makedirs("outputs", exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

# === Helper : Grid Visualization ===
def draw_grid(img, step=50, color=(0, 255, 0)):
    h, w = img.shape[:2]
    grid = img.copy()
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), color, 1)
    return grid

def save_side_by_side(title, before, after, label_text):
    h, w = before.shape[:2]
    combined = np.hstack((before, after))
    border_h = 40
    white_border = np.ones((border_h, combined.shape[1], 3), dtype=np.uint8) * 255
    final_img = np.vstack((combined, white_border))
    cv2.putText(final_img, label_text, (10, h + border_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    filename = f"outputs/output_{title}.png"
    cv2.imwrite(filename, img_as_ubyte(final_img))
    print(f"Saved: {filename}")

# === Helper: Visualize Matrix ===
def visualize_matrix(M, title, filename):
    fig = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(fig, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset = 80
    for i, row in enumerate(M):
        text = " ".join([f"{v:8.4f}" for v in row])
        cv2.putText(fig, text, (20, y_offset + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.imwrite(filename, fig)
    print(f"Matrix visualization saved: {filename}")

# === Apply affine ===
def apply_affine(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[70, 70], [220, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, M, (w, h))
    inv = cv2.invertAffineTransform(M)
    return result, M, inv

# === Apply perspective ===
def apply_perspective(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32([[50, 50], [w - 100, 30], [30, h - 50], [w - 70, h - 80]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, M, (w, h))
    inv = np.linalg.inv(M)
    return result, M, inv

def save_matrices_to_csv(all_matrices, filename):
    rows = []
    for img_name, mats in all_matrices.items():
        for mat_type, M, Minv in mats:
            for i, row in enumerate(M):
                rows.append([img_name, mat_type, f"Forward_row_{i+1}"] + row.tolist())
            for i, row in enumerate(Minv):
                rows.append([img_name, mat_type, f"Inverse_row_{i+1}"] + row.tolist())
    max_len = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_len:
            r.append("")
    df = pd.DataFrame(rows, columns=["Image", "Type", "Matrix_Row"] + [f"param_{i+1}" for i in range(max_len - 3)])
    df.to_csv(filename, index=False)
    print(f"Saved matrix parameters to {filename}\n")

# === Run experiment ===
def run_experiment(dataset_type, transform_type):
    timestamp = get_timestamp()
    all_matrices = {}

    if dataset_type == "preset":
        datasets = {
            "camera": data.camera(),
            "coins": data.coins(),
            "checkerboard": data.checkerboard(),
            "astronaut": color.rgb2gray(data.astronaut()),
            "chelsea": color.rgb2gray(data.chelsea())
        }
        print("\nAvailable preset datasets:")
        for i, k in enumerate(datasets.keys(), start=1):
            print(f"{i}. {k}")
        print(f"{len(datasets)+1}. all")
        choice = input("Choose dataset [number/comma/all]: ").strip()
        if choice.lower() == "all" or choice == str(len(datasets)+1):
            selected = datasets
        else:
            idxs = [int(i.strip()) for i in choice.split(",") if i.strip().isdigit()]
            selected = {list(datasets.keys())[i-1]: list(datasets.values())[i-1] for i in idxs}
    else:
        personal_dir = "personal"
        selected = {}
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
                selected[os.path.splitext(filename)[0]] = img_as_ubyte(img)

    for name, img in selected.items():
        if img.dtype != np.uint8:
            img = img_as_ubyte(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        grid_img = draw_grid(img)
        print(f"\nProcessing image: {name}")

        mats = []

        if transform_type in ["affine", "all"]:
            result, M, Minv = apply_affine(grid_img)
            visualize_matrix(M, f"Affine Matrix - {name}", f"outputs/matrix_affine_{name}_{timestamp}.png")
            save_side_by_side(f"{timestamp}_{dataset_type}_{name}_affine",
                              grid_img, result, f"{name}: Affine Transform (Before | After)")
            mats.append(("Affine", M, Minv))

        if transform_type in ["perspective", "all"]:
            result, M, Minv = apply_perspective(grid_img)
            visualize_matrix(M, f"Perspective Matrix - {name}", f"outputs/matrix_perspective_{name}_{timestamp}.png")
            save_side_by_side(f"{timestamp}_{dataset_type}_{name}_perspective",
                              grid_img, result, f"{name}: Perspective Transform (Before | After)")
            mats.append(("Perspective", M, Minv))

        all_matrices[name] = mats

    save_matrices_to_csv(all_matrices, f"outputs/matrix_parameters_{dataset_type}_{transform_type}_{timestamp}.csv")
    print("Experiment complete.\n")

# === Interactive Menu ===
def main():
    print("=== Geometric Transformation & Calibration Simulation ===")
    print("1. Preset Dataset (cameraman, coins, etc.)")
    print("2. Personal Photos (in 'personal/' folder)")
    dataset_choice = input("Choose dataset type [1/2]: ").strip()
    dataset_type = "preset" if dataset_choice == "1" else "personal"

    print("\nSelect transformation type:")
    print("1. Affine Transformation")
    print("2. Perspective Transformation")
    print("3. Both (All)")
    transform_choice = input("Choose transform [1/2/3]: ").strip()

    transform_type = {
        "1": "affine",
        "2": "perspective",
        "3": "all"
    }.get(transform_choice, None)

    if transform_type:
        run_experiment(dataset_type, transform_type)
    else:
        print("Invalid choice. Exiting.")

# === Entry Point ===
if __name__ == "__main__":
    main()
