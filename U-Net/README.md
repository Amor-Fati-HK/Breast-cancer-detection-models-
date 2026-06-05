# Standard U-Net Baseline

This directory contains the implementation and experimental notebook of our standard U-Net baseline. It serves as our pure visual reference point (unimodal) for both datasets.

---

## Architecture Summary
The standard U-Net (Ronneberger, 2015) uses a symmetric U-shaped design:
* **Contracting Path (Encoder):** Successive 3x3 convolutions and max-pooling to extract abstract features ("what").
* **Expanding Path (Decoder):** Up-convolutions to reconstruct spatial resolution ("where").
* **Skip Connections:** Directly transfers high-resolution features from the encoder to the decoder to preserve local spatial details.

---

## Performance Metrics

### 1. Ultrasound (BUSI Dataset)
* **Dice Coefficient:** 0.7138
* **Recall (Sensitivity):** 0.6279
* **Precision:** 0.8513
* **Specificity:** 0.9909
* **Accuracy:** 0.9625

### 2. Mammography (CBIS-DDSM Dataset)
* **Dice Coefficient:** 0.8248
* **Recall (Sensitivity):** 0.9357
* **Precision:** 0.8287
* **Specificity:** 0.6716
* **Accuracy:** 0.8366

---

## Key Limitations
* **Excessive Smoothing:** Lacks selective attention, causing the decoder to over-smooth irregular borders (like spicules).
* **Average Mask Bias:** On centered ROI crops, the model suffers from center-bias, tending to predict a generic circular or octagonal shape regardless of actual pathological contours.

