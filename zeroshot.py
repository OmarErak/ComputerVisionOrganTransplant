import os
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
CSV_PATH = 'Total_WSI.csv'
FEATURE_DIR = '../features-20251122T053023Z-1-001/features/20x_512px_0px_overlap/features_conch_v15' 
OUTPUT_CSV = 'titan_steatosis_predictions.csv'

# --- CLASS MAPPING ---
# Label 0 = High Steatosis (>30%)
# Label 1 = Low Steatosis (<30%)
CLASS_NAMES = ["High Steatosis (>30%)", "Low Steatosis (<30%)"]

# --- PROMPTS ---
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. LOAD MODEL ---
print("\nLoading TITAN model...")
model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
model = model.to(device)
model.eval()

# --- 2. SETUP CLASSIFIER ---
print("Generating classifier weights...")
prompts_list = [STEATOSIS_PROMPTS[k] for k in CLASS_NAMES]

try:
    from titan.utils import TEMPLATES
except ImportError:
    TEMPLATES = ["it is {}", "a photo of {}", "a histology slide of {}"]

with torch.autocast('cuda', torch.float16), torch.inference_mode():
    classifier = model.zero_shot_classifier(prompts_list, TEMPLATES, device=device)

# --- 3. BATCH PROCESSING ---
print(f"\nReading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

y_true = []
y_pred = []
y_scores = [] 
missing_files = []

print(f"Found {len(df)} slides. Starting inference...\n")

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Slides"):
    slide_id = row['Tissue Sample ID']
    true_label = int(row['Label']) 
    
    feature_path = os.path.join(FEATURE_DIR, f"{slide_id}.h5")
    
    if not os.path.exists(feature_path):
        missing_files.append(slide_id)
        continue
        
    try:
        with h5py.File(feature_path, 'r') as f:
            features = torch.from_numpy(f['features'][:])
            if 'coords' in f:
                coords = torch.from_numpy(f['coords'][:])
                patch_size_lv0 = f['coords'].attrs.get('patch_size_level0', 512)
            else:
                print(f"Skipping {slide_id}: No coordinates found.")
                continue

        features = features.to(device)
        coords = coords.to(device)

        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
            scores = model.zero_shot(slide_embedding, classifier)
            
            scores_vec = scores.flatten().float().cpu()
            predicted_idx = scores_vec.argmax().item()
            
        y_true.append(true_label)
        y_pred.append(predicted_idx)
        y_scores.append(scores_vec.tolist())

    except Exception as e:
        print(f"\nError processing {slide_id}: {e}")
        continue

# --- 4. METRICS & PLOTTING ---
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

if len(y_true) == 0:
    print("No slides were processed successfully.")
    exit()

# Text Report
acc = accuracy_score(y_true, y_pred)
print(f"Total Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion Matrix Logic
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix Counts:")
print(f"True High / Pred High: {cm[0][0]}")
print(f"True High / Pred Low:  {cm[0][1]}")
print(f"True Low  / Pred High: {cm[1][0]}")
print(f"True Low  / Pred Low:  {cm[1][1]}")

# --- NEW: PLOT CONFUSION MATRIX ---
plt.figure(figsize=(8, 6))
# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES,
            annot_kws={"size": 14}) # Make numbers larger

plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Zero-Shot Confusion Matrix (Acc: {acc:.1%})', fontsize=14)
plt.tight_layout()

# Save image
save_path = 'zeroshot_confusion_matrix.png'
plt.savefig(save_path, dpi=300)
print(f"\n Confusion matrix plot saved to: {save_path}")

# --- 5. SAVE CSV ---
# Align lists and save
processed_ids = []
for index, row in df.iterrows():
    if row['Tissue Sample ID'] not in missing_files:
        processed_ids.append(row['Tissue Sample ID'])

output_df = pd.DataFrame({
    'Tissue Sample ID': processed_ids,
    'True Label': y_true,
    'Predicted Label': y_pred,
    'Predicted Class': [CLASS_NAMES[i] for i in y_pred],
    'Score_High': [s[0] for s in y_scores],
    'Score_Low': [s[1] for s in y_scores]
})

output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Detailed predictions saved to: {OUTPUT_CSV}")