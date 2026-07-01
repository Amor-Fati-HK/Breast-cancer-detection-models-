# Interactive Radiology Assistant - Presentation Demo

This directory contains the final prototype and the unified demonstration notebook used for our graduation defense. 

The entire pipeline, including data loading, model inference, and the interactive web interface, is designed to be executed directly from a single Jupyter Notebook.

---

## How to Run the Demo
1. Open the Jupyter Notebook file named `presentation(1).ipynb`.
2. Execute all cells sequentially from top to bottom.
3. The final cell will launch the local Gradio server and generate an active URL (e.g., `http://127.0.0.1:7860`) to open the web interface in your browser.

---

## Crucial Step: Path Configuration
Before running the notebook cells, you must update the absolute path variables at the beginning of the script to match your local directory structure:
* **`MODEL_DIR`**: Update this variable to point to the folder containing your saved `.keras` and `.pth` weight files.
* **`base_locale`**: Update this variable to point to your local `CBIS-DDSM` dataset directory.
* **`val_df` / `val_input_imgs`**: Ensure these validation structures are loaded correctly in your active Jupyter session.

---

## How the Gradio Interface Works
The application provides an interactive, unified interface to compare all model variants across both datasets:
* **Dataset Selection:** Use the radio buttons to dynamically switch between the BUSI (Ultrasound) and CBIS-DDSM (Mammography) datasets.
* **Model Variant Dropdown:** Select any trained baseline (U-Net, CBAM-UNet, VGG19) or hybrid multimodal model (ClinicalBERT, CLIPSeg). The dropdown list updates automatically based on the chosen dataset.
* **Patient Index Slider:** Drag the slider to scroll through the validation cohort. The interface instantly loads the original scan, the radiologist's ground truth mask, the AI-predicted mask, and a semi-transparent red overlay for visual comparison.
* **Clinical Report Box:** For multimodal models, this box automatically displays the associated clinical text prompt used to guide the segmentation.
