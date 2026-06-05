# CBAM-UNet Baseline

This directory contains the implementation and experimental notebook of the CBAM-UNet baseline. It introduces spatial and channel attention mechanisms to the standard U-Net architecture.

---

## Architecture Summary
The architecture integrates the Convolutional Block Attention Module (CBAM) (Woo et al., 2018) sequentially at the bottleneck and skip connections:
* **Channel Attention:** Recalibrates feature maps to highlight pathologically relevant textures and ignore noise channels.
* **Spatial Attention:** Generates a spatial activation map to focus exclusively on the lesion region while suppressing healthy tissue.

---

## Performance Metrics

### 1. Ultrasound (BUSI Dataset)
* **Dice Coefficient:** 0.7538
* **Recall (Sensitivity):** 0.6474
* **Precision:** 0.8317
* **Specificity:** 0.9894
* **Accuracy:** 0.9623

### 2. Mammography (CBIS-DDSM Dataset)
* **Dice Coefficient:** 0.8910
* **Recall (Sensitivity):** 0.9533
* **Precision:** 0.8402
* **Specificity:** 0.6904
* **Accuracy:** 0.8554

---

## Key Achievements and Limitations
* **Contrast Improvement:** Successfully resolves false negatives on low-contrast lesions by forcing the network to focus on suspect regions.
* **Geometric Limitations:** Despite better global reconstruction, the model still struggles to define non-rounded boundaries and complex spiculated margins.
