
import pickle
import numpy as np
from config import PATHS

def main():
    print("🔹 BLOCK 3: GRAY-LEVEL ANALYSIS")
    
    # 1. Load Block Data
    try:
        with open(PATHS['multipixel_pkl'], 'rb') as f:
            blocks_data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{PATHS['multipixel_pkl']} not found. Run Step 2 first.")
    
    analyzed_blocks = []
    
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    priority_counts = {'High': 0, 'Medium': 0, 'Low': 0}

    # FIXED: Define image dimensions for 256x256 image
    height = 256
    width = 256


    # Smart Quantization using Error Diffusion (Floyd-Steinberg)
    # We can only support 3 distinct physical stick levels with 2x2 constant-weight shares:
    # Level 0: Stack Weight 4 (Black) -> Int 0
    # Level 1: Stack Weight 3 (Gray)  -> Int 1
    # Level 2: Stack Weight 2 (White) -> Int 2
    
    # We will dither the image to these 3 values.
    # Map input 0-255 to localized thresholds
    
    img_float = np.zeros((height//2, width//2), dtype=np.float32)
    contrast_grid = np.zeros((height//2, width//2), dtype=np.float32)

    # 1. Extract block averages first
    idx = 0
    for r in range(0, height, 2):
        for c in range(0, width, 2):
            img_float[r//2, c//2] = blocks_data[idx]['avg_gray']
            contrast_grid[r//2, c//2] = blocks_data[idx]['contrast_weight']
            idx += 1
            
    h_b, w_b = img_float.shape
    quantized_grid = np.zeros_like(img_float, dtype=np.uint8)
    
    # 2. Apply Dithering
    for y in range(h_b):
        for x in range(w_b):
            old_pixel = img_float[y, x]
            
            # Find nearest level
            # We now support 4 levels for higher fidelity:
            # Level 0 (Black) -> 0
            # Level 1 (Dark)  -> 85
            # Level 2 (Light) -> 170
            # Level 3 (White) -> 255
            
            val0, val1, val2, val3 = 0, 85, 170, 255
            
            dist0 = abs(old_pixel - val0)
            dist1 = abs(old_pixel - val1)
            dist2 = abs(old_pixel - val2)
            dist3 = abs(old_pixel - val3)
            
            min_dist = min(dist0, dist1, dist2, dist3)
            
            if min_dist == dist0:
                new_pixel = val0
                level = 0
            elif min_dist == dist1:
                new_pixel = val1
                level = 1
            elif min_dist == dist2:
                new_pixel = val2
                level = 2
            else:
                new_pixel = val3
                level = 3
                
            quantized_grid[y, x] = level
            quant_error = old_pixel - new_pixel
            
            # Distribute error (Floyd-Steinberg - Classic)
            #       X   7
            #   3   5   1
            # ( / 16 )
            if x + 1 < w_b:
                img_float[y, x + 1] += quant_error * 7 / 16
            if y + 1 < h_b:
                if x - 1 >= 0:
                    img_float[y + 1, x - 1] += quant_error * 3 / 16
                img_float[y + 1, x] += quant_error * 5 / 16
                if x + 1 < w_b:
                    img_float[y + 1, x + 1] += quant_error * 1 / 16
                    
    # 3. Store result back to list structure
    idx = 0
    for r in range(0, height, 2):
        for c in range(0, width, 2):
            level = quantized_grid[r//2, c//2]
            priority = 'Low'
            contrast = contrast_grid[r//2, c//2]
            
            if contrast > 0.7: priority = 'High'
            elif contrast >= 0.4: priority = 'Medium'
            
            level_counts[level] += 1
            priority_counts[priority] += 1
            
            analysis = {
                'row_idx': r,
                'col_idx': c,
                'gray_level': level,
                'priority': priority,
                'contrast_weight': contrast
            }
            analyzed_blocks.append(analysis)
            idx += 1
        
    # 4. Save Output
    with open(PATHS['gray_level_pkl'], 'wb') as f:
        pickle.dump(analyzed_blocks, f)
        
    print(f"Processed {len(analyzed_blocks)} blocks.")
    print("Gray Level Distribution:", level_counts)
    print("Priority Distribution:", priority_counts)
    print(f"Saved {PATHS['gray_level_pkl']}")

if __name__ == "__main__":
    main()
