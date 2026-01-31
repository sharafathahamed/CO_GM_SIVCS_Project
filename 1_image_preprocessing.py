
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import PATHS

def main():
    print("🔹 BLOCK 1: IMAGE PREPROCESSING")
    
    # 1. Load Image (Auto-detect)
    try:
        input_path = PATHS['input_image']()
        print(f"Processing Input Image: {input_path}")
    except FileNotFoundError as e:
        print(e)
        return

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {input_path}")
    
    # 2. Resize to 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # 3. Edge-Preserving Smoothing (Bilateral Filter)
    filtered = cv2.bilateralFilter(img, 9, 75, 75)

    # 4. AHP-style Normalization
    p2, p98 = np.percentile(filtered, (2, 98))
    stretched = np.clip((filtered - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

    # 5. CLAHE + Gamma Correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)
    
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    # Save Enhanced Image
    cv2.imwrite(PATHS['enhanced_image'], enhanced)
    print(f"Saved {PATHS['enhanced_image']}")

    # 6. Edge-Aware Contrast Map
    edges = cv2.Canny(enhanced, 100, 200)
    edges_blurred = cv2.GaussianBlur(edges, (5, 5), 0)
    
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    contrast_map_raw = cv2.addWeighted(edges_blurred, 0.5, magnitude, 0.5, 0)
    contrast_map = contrast_map_raw.astype(np.float32) / 255.0
    
    # Save Contrast Map (intermediate data)
    cv2.imwrite(PATHS['contrast_map'], (contrast_map * 255).astype(np.uint8))
    print(f"Saved {PATHS['contrast_map']}")

if __name__ == "__main__":
    main()

