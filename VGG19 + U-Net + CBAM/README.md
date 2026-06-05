# VGG19-UNet with CBAM

This directory contains the implementation and experimental notebook of the VGG19-UNet with CBAM attention. It leverages transfer learning with a pre-trained deep feature extractor to improve boundary segmentation.

---

## Architecture and Training Summary
The architecture replaces the standard U-Net contracting path with a deep, pre-trained backbone:
* **VGG19 Encoder:** Uses weights pre-trained on ImageNet to extract robust, deep texture features from day one.
* **CBAM Integration:** An attention module recalibrates deep feature maps at the bottleneck.
* **Two-Phase Training:** 
  1. *Warm-up Phase:* The VGG19 encoder is frozen. Only the decoder and attention layers are trained.
  2. *Fine-Tuning Phase:* Deep convolutional blocks (blocks 4 & 5) are unfrozen with a low learning rate (1e-5) to adapt to radiological textures.

---

## Performance Metrics

### 1. Ultrasound (BUSI Dataset)
* **Dice Coefficient:** 0.7337
* **Recall (Sensitivity):** 0.7117
* **Precision:** 0.7720
* **Specificity:** 0.9822
* **Accuracy:** 0.9612

### 2. Mammography (CBIS-DDSM Dataset)
* **Dice Coefficient:** 0.8486
* **Recall (Sensitivity):** 0.9148
* **Precision:** 0.8581
* **Specificity:** 0.7065
* **Accuracy:** 0.8447

---

## Key Achievements and Limitations
* **Stable Textures:** Transfer learning significantly improves the detection of raw geometric structures.
* **Conservative Trade-off:** The model is more conservative, increasing specificity (fewer false positives) but occasionally missing peripheral micro-extensions of the tumors.
