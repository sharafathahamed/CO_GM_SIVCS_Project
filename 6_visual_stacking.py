
import cv2
import numpy as np
from config import PATHS

def main():
    print("🔹 BLOCK 6: VISUAL STACKING")
    
    share1 = cv2.imread(PATHS['share1'], cv2.IMREAD_GRAYSCALE)
    share2 = cv2.imread(PATHS['share2'], cv2.IMREAD_GRAYSCALE)
    
    if share1 is None or share2 is None:
        raise FileNotFoundError("Shares not found. Run Step 5 first.")
        
    # Simulate Physical Stacking (OR operation on BLACK/INK pixels)
    stacked = cv2.bitwise_and(share1, share2)
    
    cv2.imwrite(PATHS['stacked'], stacked)
    print(f"Saved {PATHS['stacked']}")
    print("Visual Stacking Complete.")

if __name__ == "__main__":
    main()
