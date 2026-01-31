
import cv2
import numpy as np
import pickle
from config import PATHS

def main():
    print("🔹 BLOCK 2: MULTIPIXEL BLOCK FORMATION")
    
    # 1. Load Enhanced Image and Contrast Map
    img = cv2.imread(PATHS['enhanced_image'], cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{PATHS['enhanced_image']} not found. Run Step 1 first.")
    
    contrast_img = cv2.imread(PATHS['contrast_map'], cv2.IMREAD_GRAYSCALE)
    if contrast_img is None:
        raise FileNotFoundError(f"{PATHS['contrast_map']} not found. Run Step 1 first.")
    contrast_map = contrast_img.astype(np.float32) / 255.0

    height, width = img.shape
    if height != 256 or width != 256:
        raise ValueError("Image must be 256x256")

    # 2. Divide into 2x2 blocks
    block_size = 2
    blocks_data = []
    
    for r in range(0, height, block_size):
        for c in range(0, width, block_size):
            block_pixels = img[r:r+block_size, c:c+block_size]
            contrast_block = contrast_map[r:r+block_size, c:c+block_size]
            
            block_info = {
                'row_idx': r,
                'col_idx': c,
                'pixels': block_pixels,
                'avg_gray': np.mean(block_pixels),
                'contrast_weight': np.mean(contrast_block),
                'range': np.max(block_pixels) - np.min(block_pixels),
                'variance': np.var(block_pixels)
            }
            blocks_data.append(block_info)

    total_blocks = len(blocks_data)
    print(f"Total blocks processed: {total_blocks}")

    # 3. Store in multipixel_output.pkl
    with open(PATHS['multipixel_pkl'], 'wb') as f:
        pickle.dump(blocks_data, f)
    
    print(f"Saved {PATHS['multipixel_pkl']}")

if __name__ == "__main__":
    main()
