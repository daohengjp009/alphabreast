# AlphaBreast

**Multi-View Attention Network for Breast Cancer Detection Inspired by AlphaFold's Evoformer Architecture**

CM3070 Computer Science Final Project  
Author: YU KAM LO  
Date: March 2026

## Overview

AlphaBreast is a deep learning system that classifies mammograms as malignant or benign by integrating paired Craniocaudal (CC) and Mediolateral Oblique (MLO) views through attention mechanisms inspired by AlphaFold's Evoformer architecture. The system uses a shared Swin Transformer backbone with Evoformer-inspired cross-attention blocks, evaluated on CBIS-DDSM with 5-fold patient-level cross-validation.

## Version History

| Version | Architecture | Data | Accuracy | AUC |
|---------|-------------|------|----------|-----|
| V1 | 3-layer CNN, Single Attention | Mass only (~509 pairs) | 51.0% | 0.614 |
| V2 | ResNet18, Bidirectional Attention, Focal Loss | Mass only (~509 pairs) | 60.3% | 0.720 |
| V3 | ResNet18, Bidirectional Attention | Mass + Calc (~1000 pairs) | 68.0% | 0.740 |
| V4 | Swin Transformer, Evoformer blocks, 5-Fold CV | Mass + Calc (1,062 pairs) | 72.3% +/- 4.7% | 0.799 +/- 0.050 |

## Repository Contents

```
AlphaBreast_V1.ipynb          - Baseline: 3-layer CNN + single attention
AlphaBreast_V2.ipynb          - ResNet18 + bidirectional attention + Focal Loss
AlphaBreast_V3.ipynb          - V2 architecture + combined mass/calc data
AlphaBreast_V4_Colab.ipynb    - Final: Swin Transformer + Evoformer + 5-fold patient-level CV
gradcam_cell.py               - Grad-CAM visualisation (paste into V4 notebook after training)
README.md                     - This file
```

## Dataset

This project uses the **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) dataset.

**The dataset is NOT included in this repository** due to its size.

To reproduce results:

1. Download CBIS-DDSM from Kaggle: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
2. Upload to Google Drive with this structure:
   ```
   Google Drive/
     CBIS-DDSM/
       csv/
         mass_case_description_train_set.csv
         mass_case_description_test_set.csv
         calc_case_description_train_set.csv
         calc_case_description_test_set.csv
       jpeg/
         1.3.6.1.4.1.9590.100.1.../
           1-207.jpg
           2-208.jpg
         ...
   ```
3. Update `DATA_ROOT` in the notebook if your path differs from `/content/drive/MyDrive/CBIS-DDSM`

## Requirements

- Python 3.10+
- PyTorch 2.0+
- timm (PyTorch Image Models)
- scikit-learn
- Google Colab with T4 GPU (recommended)

Install dependencies:
```bash
pip install timm
```

## Running the Code

1. Open any notebook in Google Colab
2. Enable GPU runtime: Runtime > Change runtime type > T4 GPU
3. Mount Google Drive when prompted
4. Run all cells

V4 training with 5-fold CV for all 4 model configurations takes approximately 10-12 hours on a T4 GPU.

## Key Features

- Swin Transformer backbone (pretrained on ImageNet-22K)
- Evoformer-inspired attention blocks with bidirectional cross-attention, pairwise updates, and gated FFN
- Patient-level GroupKFold cross-validation to prevent data leakage
- Separated augmentation pipeline: shared geometric transforms, independent intensity transforms
- Grad-CAM visualisation for interpretability

## Project Template

CM3015 Machine Learning and Neural Networks - Project Idea 2: Deep Learning Breast Cancer Detection

## License

This project is submitted as coursework for CM3070 at the University of London. The code is provided for academic review and reproducibility purposes.
