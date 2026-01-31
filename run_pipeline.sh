#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting CO GM-SIVCS Pipeline..."

echo "[1/8] Running Image Preprocessing..."
python3 1_image_preprocessing.py

echo "[2/8] Running Multipixel Block Formation..."
python3 2_multipixel_blocks.py

echo "[3/8] Running Gray-Level Analysis..."
python3 3_gray_level_analysis.py

echo "[4/8] Running CO GM-SIVCS Encoding..."
python3 4_co_gm_sivcs_encoding.py

echo "[5/8] Running Share Generation..."
python3 5_share_generation.py

echo "[6/8] Running Visual Stacking Simulation..."
python3 6_visual_stacking.py

echo "[7/8] Running Computational Reconstruction..."
python3 7_reconstruction.py

echo "[8/8] Running Performance Evaluation..."
python3 8_performance_evaluation.py

echo "------------------------------------------------"
echo "✅ Pipeline Completed Successfully!"
echo "Outputs generated:"
echo " - Shares: Share1.png, Share2.png"
echo " - Reconstruction: Stacked.png, Reconstructed.png"
echo " - Metrics: Check terminal output above"
