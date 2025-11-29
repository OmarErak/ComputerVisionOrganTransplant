# Organ Transplant Decision Based on Liver Steatosis Classification

This repository contains the code for our project on **automated liver steatosis assessment** from Whole Slide Images (WSIs) using pathology foundation models.  

The pipeline combines:

- **TRIDENT** for scalable WSI processing and patch-level feature extraction (CONCH v1.5). :contentReference[oaicite:0]{index=0}  
- **TITAN** for slide-level multimodal feature analysis, zero-shot classification, and slide embeddings. :contentReference[oaicite:1]{index=1}  

We predict whether each liver WSI has **High steatosis (>30%)** or **Low steatosis (<30%)**, and compare:

- **Zero-shot TITAN classification** (no training labels)  
- **Supervised MLP classifier** trained on frozen TITAN slide embeddings  

---

## Repository Structure

```text
.
├── feature_extraction.py   # TRIDENT-based WSI → CONCH v1.5 patch features (.h5)
├── zeroshot.py             # TITAN zero-shot steatosis classification
├── supervised.py           # Supervised MLP on TITAN slide embeddings
├── Total_WSI.csv           # Metadata: slide IDs, labels, pathology notes, etc.
└── README.md
````

---

## Data

### Total_WSI.csv

`Total_WSI.csv` contains slide-level metadata used across scripts. At minimum it is expected to include:

* `Tissue Sample ID` – unique identifier for each WSI
* `Label` – steatosis label (e.g. `0 = High`, `1 = Low` in the code)
* Optional columns – e.g. free-text **pathology notes** or reports used for manual labeling

Your WSIs themselves are **not** stored in this repository; you should keep them in your own storage and point `feature_extraction.py` to that directory.

---

## Dependencies

* Python ≥ 3.9
* PyTorch ≥ 2.0
* `transformers`
* `huggingface_hub`
* `h5py`, `pandas`, `numpy`, `scikit-learn`
* `tqdm`, `matplotlib` (optional, for plots)
* **TRIDENT** (installed from source) ([GitHub][1])
* **TITAN** (installed from source + Hugging Face access) ([GitHub][2])

> ⚠️ **Medical use disclaimer:** TRIDENT, CONCH, and TITAN are released for **non-commercial research only** under their respective licenses. This project is a research prototype and **not** a clinical decision support tool.

---

## 1. Environment Setup

### 1.1 Clone this repository

```bash
git OmarErak/ComputerVisionOrganTransplant.git
cd OmarErak/ComputerVisionOrganTransplant
```

### 1.2 Install a base environment

You can adapt this to your preferred environment manager:

```bash
conda create -n liver-titan python=3.9 -y
conda activate liver-titan

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install transformers huggingface_hub h5py pandas numpy scikit-learn tqdm matplotlib
```

---

## 2. Install TRIDENT

TRIDENT is a toolkit from MahmoodLab for WSI preprocessing and feature extraction. ([GitHub][1])

```bash
# From somewhere outside this repo:
git clone https://github.com/mahmoodlab/TRIDENT.git
cd TRIDENT
pip install -e .
cd ..
```

TRIDENT provides:

* Tissue segmentation
* Patch extraction
* Patch feature extraction using **CONCH v1.5** and other foundation models
* Slide-level feature extraction utilities

---

## 3. Install TITAN

TITAN is a multimodal whole-slide foundation model trained on CONCH v1.5 patch features.

```bash
git clone https://github.com/mahmoodlab/TITAN.git
cd TITAN
pip install -e .
cd ..
```

### 3.1 Request access to TITAN and CONCH v1.5 weights

Both TITAN and CONCH v1.5 are hosted on Hugging Face and require **gated access**:

* TITAN model card: `https://huggingface.co/MahmoodLab/TITAN` 
* CONCH v1.5 model card: `https://huggingface.co/MahmoodLab/conchv1_5` 

Follow the instructions on each model page to **request access with an institutional email**.

### 3.2 Login to Hugging Face

```python
from huggingface_hub import login

login()  # paste your HF token from https://huggingface.co/settings/tokens
```
 
---

## 4. Step 1 – Patch Feature Extraction (`feature_extraction.py`)

This script uses **TRIDENT** to:

1. Read WSIs from a directory
2. Perform tissue detection / patch extraction
3. Extract **CONCH v1.5** patch embeddings (512×512 at 20×)
4. Save features and coordinates to HDF5 (`.h5`) files compatible with TITAN ([GitHub][2])

**Example :**

```bash
python feature_extraction.py \
    --wsi_dir /path/to/wsis \
    --output_dir /path/to/features \
    --patch_size 512 \
    --magnification 20 \
    --model_name conch_v1_5
```

Typical output structure:

```text
/path/to/features/
├── <SLIDE_ID_1>.h5
├── <SLIDE_ID_2>.h5
└── ...
```

Each `.h5` file should contain:

* `features` – [num_patches, 768] CONCH patch embeddings
* `coords`   – [num_patches, 2] patch coordinates
* `coords.attrs["patch_size_level0"]` – distance between adjacent patches at level 0

---

### 5. Step 2 – Zero-Shot Classification (`zeroshot.py`)

`zeroshot.py` is a **self-contained script** that uses TITAN to run zero-shot steatosis classification on all slides listed in `Total_WSI.csv`.

At the top of the file there is a **configuration block**:

```python
# --- CONFIGURATION ---
CSV_PATH = 'Total_WSI.csv'
FEATURE_DIR = '../features-20251122T053023Z-1-001/features/20x_512px_0px_overlap/features_conch_v15'
OUTPUT_CSV = 'titan_steatosis_predictions.csv'

# Label mapping and class names
# Label 0 = High Steatosis (>30%)
# Label 1 = Low Steatosis (<30%)
CLASS_NAMES = ["High Steatosis (>30%)", "Low Steatosis (<30%)"]

# Text prompts for zero-shot classification (prompt ensemble)
STEATOSIS_PROMPTS = {
    "High Steatosis (>30%)": [
        "A histology slide dominated by large white fat vacuoles.",
        "Severe fatty change with crowded clear circular spaces.",
        "Macrovesicular steatosis replacing more than half the tissue.",
        "Pathology image showing abundant lipid droplets displacing nuclei.",
        "Swiss cheese texture with very little solid pink tissue."
    ],
    "Low Steatosis (<30%)": [
        "A histology slide dominated by solid pink cytoplasm.",
        "Preserved hepatic architecture with no significant fat vacuoles.",
        "Dense eosinophilic tissue without many white holes.",
        "Intact parenchyma consisting of uniform hepatocytes.",
        "Tissue showing mostly nuclei and cytoplasm, not fat."
    ]
}
```

To run zero-shot inference you only need to:

1. Edit `CSV_PATH`, `FEATURE_DIR`, and `OUTPUT_CSV` to match your setup.
2. Optionally edit `STEATOSIS_PROMPTS` if you want to experiment with different prompts.
3. Run:

```bash
python zeroshot.py
```

What the script does:

* Loads TITAN from Hugging Face.
* Builds a zero-shot classifier from the prompt ensemble.
* For each slide ID in `Total_WSI.csv`:

  * Loads the corresponding `.h5` feature file from `FEATURE_DIR`.
  * Computes the TITAN slide embedding.
  * Applies zero-shot classification (High vs Low steatosis).
* Prints:

  * Accuracy, precision, recall, F1-score.
  * Confusion matrix counts.
* Saves:

  * A **PNG confusion matrix** (`zeroshot_confusion_matrix.png`).
  * A detailed **per-slide CSV** (`OUTPUT_CSV`) containing:

    * `Tissue Sample ID`
    * `True Label`
    * `Predicted Label`
    * `Predicted Class`
    * `Score_High`, `Score_Low` (raw scores / logits per class).

---

### 6. Step 3 – Supervised Classification (`supervised.py`)

`supervised.py` has a similar style: configuration is done via **global variables at the top of the file**, and then you simply run the script.

A typical configuration block looks like:


To run supervised training:

1. Edit `CSV_PATH`, `FEATURE_DIR`, and `OUTPUT_DIR` to match your environment.
2. Optionally adjust training hyperparameters (learning rate, dropout, batch size, etc.).
3. Run:

```bash
python supervised.py
```

What the script does:

* Loads TITAN and computes (or loads) slide embeddings from the `.h5` feature files.
* Joins embeddings with labels from `Total_WSI.csv`.
* Trains an MLP classifier on frozen TITAN embeddings using **stratified K-fold cross-validation** (e.g. 5 folds).
* Optionally runs Bayesian optimization inside the script to refine LR / dropout / weight decay.
* Outputs:

  * Console metrics per fold and the mean accuracy / precision / recall / F1.
  * Confusion matrices and training curves (loss/accuracy plots) saved under `OUTPUT_DIR`.
  * A CSV with per-fold metrics and, optionally, a saved model checkpoint (e.g. `best_model.pt`).

---

## 7. Reproducibility

To make your experiments reproducible:

* Fix random seeds inside `zeroshot.py` and `supervised.py` (Python, NumPy, PyTorch).
* Document which version of TRIDENT, TITAN, and PyTorch you used.
* Save:

  * The exact prompts used for zero-shot classification
  * The final hyperparameters for the supervised MLP

---

## 8. References

If you use this repository in academic work, please cite:

* **TRIDENT** – scalable WSI processing toolkit. 
* **CONCH v1.5** – vision–language pathology foundation model. 
* **TITAN** – multimodal whole-slide foundation model. 


---

## 9. Contact

For questions about this code:

TBD

For questions about TRIDENT, CONCH, or TITAN, please refer to the official MahmoodLab repositories and documentation.
