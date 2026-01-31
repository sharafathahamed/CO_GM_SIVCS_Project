
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import matplotlib.pyplot as plt
from config import PATHS

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def calculate_entropy(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    histogram = histogram.ravel() / histogram.sum()
    logs = np.log2(histogram + 0.00001)
    entropy = -1 * (histogram * logs).sum()
    return entropy

def main():
    print("🔹 BLOCK 8: PERFORMANCE EVALUATION")
    
    try:
        input_path = PATHS['input_image']()
    except FileNotFoundError:
        print("No input image found to compare against.")
        return

    original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    reconstructed = cv2.imread(PATHS['reconstructed'], cv2.IMREAD_GRAYSCALE)
    share1 = cv2.imread(PATHS['share1'], cv2.IMREAD_GRAYSCALE)
    
    if original is None or reconstructed is None:
        raise FileNotFoundError("Images missing.")

    # Ensure dimensions match for PSNR/SSIM
    if original.shape != reconstructed.shape:
        print(f"Resizing original {original.shape} to match reconstructed {reconstructed.shape}...")
        original = cv2.resize(original, (reconstructed.shape[1], reconstructed.shape[0]), interpolation=cv2.INTER_AREA)
        
    print(f"Original Shape: {original.shape}")
    print(f"Reconstructed Shape: {reconstructed.shape}")
    
    psnr_val = calculate_psnr(original, reconstructed)
    ssim_val = ssim(original, reconstructed, data_range=255)
    entropy_val = calculate_entropy(share1)
    
    print("\n" + "="*30)
    print("   PERFORMANCE METRICS")
    print("="*30)
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"Share Entropy: {entropy_val:.4f}")
    print("="*30)

    # Visual Popup
    print("\nGenerating Visual Popup...")
    enhanced = cv2.imread(PATHS['enhanced_image'], cv2.IMREAD_GRAYSCALE)
    stacked = cv2.imread(PATHS['stacked'], cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Original Input")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("2. Physical Stacking (Visual Decryption)")
    plt.imshow(stacked, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("3. Smart Reconstruction (Result)")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
