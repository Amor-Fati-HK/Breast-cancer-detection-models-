# Multimodal Hybrid (ClinicalBERT + VGG19 + CBAM)

This directory contains the implementation and experimental notebook of our first multimodal architecture. It introduces clinical text guiding the deep convolutional networks.

---

## Architecture and Fusion Summary
This dual-input model combines textual clinical reports with visual spatial features:
* **Textual Encoder (ClinicalBERT):** Pre-trained on hospital notes (MIMIC-III), it converts the clinical prompt into a 768-dimensional contextual vector.
* **Visual Encoder (VGG19):** Extracts raw visual features from the mammography.
* **Cross-Modality Attention Gates:** Placed at each skip connection and bottleneck level, these gates align and modulate visual feature maps with the text vector.
* **Hadamard Modulation (FiLM):** The text vector is projected and multiplied pixel-by-pixel with the image features, acting as a semantic filter.

---

## Performance Metrics

### Mammography (CBIS-DDSM Dataset Only)
* **Dice Coefficient:** 0.7546
* **F1-Score:** 0.7546
* **Recall (Sensitivity):** 0.8950
* **Precision:** 0.6525
* **Specificity:** 0.7941
* **Accuracy:** 0.8412

---

## Key Achievements and Limitations
* **Improved Localization:** Clinical reports successfully lock onto the general region of interest (high Recall).
* **Semantic Saturation:** Because the text vector is forced into the bottleneck of a separately trained visual encoder, it crushes the fine spatial pixels. This causes blocky segmentations and lower Dice/Precision scores.
