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

After this, both TRIDENT and TITAN will be able to download TITAN and CONCH v1.5 weights via `AutoModel.from_pretrained(...)` as in the TITAN README. 
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

## 5. Step 2 – Zero-Shot Classification (`zeroshot.py`)

`zeroshot.py` loads your **patch features** and `Total_WSI.csv`, builds TITAN slide embeddings, and performs **zero-shot steatosis classification** using prompt ensembling.

High-level steps:

1. Load TITAN from Hugging Face:

   ```python
   from transformers import AutoModel
   titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
   ```
2. For each `.h5` feature file:

   * Load `features`, `coords`, `patch_size_lv0`
   * Compute `z_slide = titan.encode_slide_from_patch_features(...)`
3. Define multiple textual prompts for:

   * **High Steatosis (>30%)**
   * **Low Steatosis (<30%)**
4. Build a zero-shot classifier from prompts
5. Compute cosine similarity between `z_slide` and text prototypes
6. Save predictions and metrics

**Example usage:**

```bash
python zeroshot.py \
    --features_dir /path/to/features \
    --csv_path Total_WSI.csv \
    --output_csv titan_zeroshot_results.csv
```

The script typically writes:

* Per-slide predictions (probabilities + predicted class)
* Summary metrics (accuracy, precision, recall, F1)

---

## 6. Step 3 – Supervised Classification (`supervised.py`)

`supervised.py` trains a small **MLP classifier** on top of frozen TITAN slide embeddings to predict high vs low steatosis.

Pipeline:

1. Load TITAN and slide embeddings (computed similarly to `zeroshot.py`)
2. Merge embeddings with labels from `Total_WSI.csv`
3. Train an MLP using **stratified 5-fold cross-validation**
4. (Optionally) use **Bayesian optimization** to tune learning rate, dropout, and weight decay
5. Report mean performance across folds and save the trained model/checkpoints

**Example usage:**

```bash
python supervised.py \
    --features_dir /path/to/features \
    --csv_path Total_WSI.csv \
    --output_dir results/supervised
```

Output might include:

* `supervised_metrics.csv` – per-fold and mean metrics
* `best_model.pt` – trained MLP weights
* Training curves (loss/accuracy plots)

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

* **TRIDENT** – scalable WSI processing toolkit. ([TRIDENT Documentation][6])
* **CONCH v1.5** – vision–language pathology foundation model. ([GitHub][7])
* **TITAN** – multimodal whole-slide foundation model. ([arXiv][8])

(And your own thesis/paper when it is published.)

---

## 9. Contact

For questions about this code:

* **Your Name(s)** – <[email@institution.edu](mailto:email@institution.edu)>

For questions about TRIDENT, CONCH, or TITAN, please refer to the official MahmoodLab repositories and documentation.
