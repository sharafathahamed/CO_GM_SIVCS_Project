
import pickle
import numpy as np
import cv2
from config import PATHS

def main():
    print("🔹 BLOCK 5: SHARE GENERATION")
    
    try:
        with open(PATHS['encoding_pkl'], 'rb') as f:
            data = pickle.load(f)
            patterns = data['patterns']
            decisions = data['decisions']
    except FileNotFoundError:
        raise FileNotFoundError(f"{PATHS['encoding_pkl']} not found. Run Step 4 first.")

    height = 256
    width = 256
    
    share1 = np.ones((height, width), dtype=np.uint8) * 255
    share2 = np.ones((height, width), dtype=np.uint8) * 255
    
    for decision in decisions:
        r = decision['row_idx']
        c = decision['col_idx']
        
        p1 = patterns[decision['share1_idx']]
        p2 = patterns[decision['share2_idx']]
        
        share1[r:r+2, c:c+2] = 255 - (p1 * 255)
        share2[r:r+2, c:c+2] = 255 - (p2 * 255)
            
    cv2.imwrite(PATHS['share1'], share1)
    cv2.imwrite(PATHS['share2'], share2)
    
    print(f"Saved {PATHS['share1']}")
    print(f"Saved {PATHS['share2']}")
    
    print(f"Share1 Mean Gray: {np.mean(share1)}")
    print(f"Share2 Mean Gray: {np.mean(share2)}")

if __name__ == "__main__":
    main()
