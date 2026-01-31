
import pickle
import numpy as np
import random
from config import PATHS

def get_patterns_weight_2():
    patterns = []
    indices = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]
    for idx_pair in indices:
        mat = np.zeros((2, 2), dtype=np.uint8)
        r1, c1 = divmod(idx_pair[0], 2)
        r2, c2 = divmod(idx_pair[1], 2)
        mat[r1, c1] = 1
        mat[r2, c2] = 1
        patterns.append(mat)
    return patterns

def main():
    print("🔹 BLOCK 4: CO GM-SIVCS ENCODING")
    
    try:
        with open(PATHS['gray_level_pkl'], 'rb') as f:
            analyzed_blocks = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{PATHS['gray_level_pkl']} not found. Run Step 3 first.")
        
    patterns = get_patterns_weight_2()
    
    valid_pairs = {2: [], 3: [], 4: []}
    
    for i in range(len(patterns)):
        for j in range(len(patterns)):
            p1 = patterns[i]
            p2 = patterns[j]
            stacked = np.bitwise_or(p1, p2)
            w = np.sum(stacked)
            if w in valid_pairs:
                valid_pairs[w].append((i, j))
                
    encoding_output = []
    
    for block in analyzed_blocks:
        level = block['gray_level']
        
        # New 4-Level High Fidelity Mapping
        # Level 0 (Black) -> Stack 4 (Total Black)
        # Level 1 (Dark)  -> Stack 3
        # Level 2 (Light) -> Stack 2
        # Level 3 (White) -> Stack 2 (We use specific 'aligned' pairs that visually look lighter if possible, else same as 2)
        # Note: In 2x2 shares with weight 2, the min stack weight is 2. We can't go lower (whiter) than weight 2 physically.
        # However, the dithering in Step 3 distributes these 'white' blocks to create the illusion of brightness.
        
        target_weight = 4
        if level == 1:
            target_weight = 3
        elif level >= 2:
            target_weight = 2
            
        options = valid_pairs[target_weight]
        selected_pair_indices = random.choice(options)
        
        encoding_output.append({
            'row_idx': block['row_idx'],
            'col_idx': block['col_idx'],
            'share1_idx': selected_pair_indices[0],
            'share2_idx': selected_pair_indices[1]
        })
        
    with open(PATHS['encoding_pkl'], 'wb') as f:
        pickle.dump({'patterns': patterns, 'decisions': encoding_output}, f)
        
    print(f"Encoded {len(encoding_output)} blocks.")
    print(f"Saved {PATHS['encoding_pkl']}")

if __name__ == "__main__":
    main()
