
import os
import glob

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder Paths
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Function to get the input image automatically
def get_input_image_path():
    # Look for common image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not files:
        raise FileNotFoundError(f"No image found in {INPUT_DIR}. Please add an image.")
    
    # Return the first found image
    return files[0]

# File Paths (Centralized)
PATHS = {
    'input_image': get_input_image_path, # Function reference
    'enhanced_image': os.path.join(OUTPUT_DIR, 'enhanced_image.png'),
    'contrast_map': os.path.join(DATA_DIR, 'contrast_map.png'), # Map is intermediate data
    'multipixel_pkl': os.path.join(DATA_DIR, 'multipixel_output.pkl'),
    'gray_level_pkl': os.path.join(DATA_DIR, 'gray_level_output.pkl'),
    'encoding_pkl': os.path.join(DATA_DIR, 'co_gm_sivcs_output.pkl'),
    'share1': os.path.join(OUTPUT_DIR, 'Share1.png'),
    'share2': os.path.join(OUTPUT_DIR, 'Share2.png'),
    'stacked': os.path.join(OUTPUT_DIR, 'Stacked.png'),
    'reconstructed': os.path.join(OUTPUT_DIR, 'Reconstructed.png')
}
