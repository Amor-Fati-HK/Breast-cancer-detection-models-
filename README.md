# Breast Lesion Segmentation: From Unimodal Baselines to Multimodal VLMs

This repository contains the complete codebase, architectures, and experimental notebooks developed for automatic breast lesion segmentation. Our research follows a progressive, pyramidal optimization strategy, transitioning from pure visual models to advanced multimodal vision-language architectures.

---

## Repository Structure
Each directory contains the specific model architecture implementation, its training Jupyter Notebook, and a dedicated local `README.md` describing its hyperparameters and performance.

```text
Projet_PFE_Models/
├── figures/                       # Contains all architecture diagrams
│   ├── unet_archi.png
│   ├── cbam_archi.png
│   ├── vgg_archi.png
│   ├── clinicalbert_archi.png
│   └── clipseg_sota_archi.png
├── main_gradio.py                 # Unified local Radiology Assistant UI
├── U-Net/                         # 1. Standard visual baseline
│   ├── model.py, model.ipynb, README.md
├── CBAM-UNet/                     # 2. Spatial & Channel attention baseline
│   ├── model.py, model.ipynb, README.md
├── VGG19-UNet/                    # 3. Transfer Learning baseline
│   ├── model.py, model.ipynb, README.md
├── ClinicalBERT/                  # 4. Naive multimodal hybrid (Feature modulation)
│   ├── model.py, model.ipynb, README.md
├── CLIPSeg-Refiner/               # 5. Final Multimodal SOTA (VLM + CNN)
│   ├── model.py, model.ipynb, README.md
├── .gitignore
└── README.md                      # This global documentation file
```

---

# Experimental Progression (Summary)

* **1. U-Net (Standard Baseline):** Our spatial baseline. Accurate localization but suffers from excessive boundary smoothing.
* **2. CBAM-UNet:** Integrates spatial and channel attention to suppress acoustic/tissue noise and improve low-contrast detection.
* **3. VGG19-UNet:** Leverages pre-trained ImageNet weights for robust, deep texture extraction.
* **4. ClinicalBERT (Naive Multimodal):** Injects clinical text representations (768-dim) at the bottleneck, introducing semantic-guided segmentation.
* **5. CLIPSeg + Deep UNet (SOTA Final):** Our final contribution. Uses a pre-aligned vision-language latent space to guide a deep convolutional decoder, precisely "sculpting" tumor spicules.

---
