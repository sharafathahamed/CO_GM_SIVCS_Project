
import cv2
import numpy as np
from config import PATHS

def main():
    print("🔹 BLOCK 7: RECONSTRUCTION")
    
    stacked = cv2.imread(PATHS['stacked'], cv2.IMREAD_GRAYSCALE)
    if stacked is None:
        raise FileNotFoundError(f"{PATHS['stacked']} not found. Run Step 6 first.")
        
    height, width = stacked.shape
    reconstructed = np.zeros((height, width), dtype=np.uint8)
    
    for r in range(0, height, 2):
        for c in range(0, width, 2):
            block = stacked[r:r+2, c:c+2]
            black_count = np.sum(block < 128)
            
            # 4-Level High-Fidelity Density Mapping
            # 4 Black -> 0
            # 3 Black -> 85
            # 2 Black -> 170 (or 255 depending on context, we map to linear)
            # 0/1 Black -> 255
            
            gray_val = 0
            if black_count >= 4:
                gray_val = 0
            elif black_count == 3:
                gray_val = 85
            elif black_count == 2:
                gray_val = 170
            else:
                gray_val = 255
                
            reconstructed[r:r+2, c:c+2] = gray_val
            
    # Smart Descreening v2 (Bilateral Filter)
    # Bilateral filter smooths the noise but keeps the edges (eyes, nose) sharp.
    # d=9, sigmaColor=75, sigmaSpace=75
    reconstructed = cv2.bilateralFilter(reconstructed, 9, 80, 80)
    
    cv2.imwrite(PATHS['reconstructed'], reconstructed)
    print(f"Saved {PATHS['reconstructed']}")

if __name__ == "__main__":
    main()
