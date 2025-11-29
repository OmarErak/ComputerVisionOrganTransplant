import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
CSV_PATH = 'Total_WSI.csv'
FEATURE_DIR = '../features-20251122T053023Z-1-001/features/20x_512px_0px_overlap/features_conch_v15' 
K_FOLDS = 5          
BATCH_SIZE = 32      
EPOCHS = 30          
LEARNING_RATE = 0.001
SEED = 37

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. MODEL DEFINITION ---
class SteatosisMLP(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):
        super(SteatosisMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(0.2), # High dropout for small data
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            #nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# --- 2. DATASET CLASS ---
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 3. DATA PREPARATION ---
def prepare_data():
    print("\nLoading TITAN model (for feature aggregation)...")
    titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    titan_model = titan_model.to(device)
    titan_model.eval()

    print(f"Reading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    X_raw = []
    y_raw = []
    slide_ids = []
    
    print("Aggregating features into slide embeddings...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        slide_id = row['Tissue Sample ID']
        label = int(row['Label'])
        
        feature_path = os.path.join(FEATURE_DIR, f"{slide_id}.h5")
        
        if not os.path.exists(feature_path):
            continue
            
        try:
            with h5py.File(feature_path, 'r') as f:
                features = torch.from_numpy(f['features'][:])
                if 'coords' in f:
                    coords = torch.from_numpy(f['coords'][:])
                    patch_size_lv0 = f['coords'].attrs.get('patch_size_level0', 512)
                else:
                    continue

            features = features.to(device)
            coords = coords.to(device)
            
            with torch.autocast('cuda', torch.float16), torch.inference_mode():
                slide_embedding = titan_model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
                embedding_vec = slide_embedding.float().cpu().numpy().flatten()
                
            X_raw.append(embedding_vec)
            y_raw.append(label)
            slide_ids.append(slide_id)

        except Exception as e:
            print(f"Error {slide_id}: {e}")
            continue

    return np.array(X_raw), np.array(y_raw), slide_ids

# --- 4. TRAINING FUNCTION (Per Fold) ---
def train_fold(fold_idx, train_idx, val_idx, X, y):
    print(f"\n--- FOLD {fold_idx+1}/{K_FOLDS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    train_dset = FeatureDataset(X_train, y_train)
    val_dset = FeatureDataset(X_val, y_val)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SteatosisMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Track history for plotting
    history = {'loss': [], 'acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] # For ROC
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Store probability of the "positive" class (Class 1: Low Steatosis, or Class 0: High)
            # Storing Class 1 probability for ROC
            all_probs.extend(probs[:, 1].cpu().numpy()) 
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"Fold {fold_idx+1} Accuracy: {acc:.2%}")
    
    return acc, all_labels, all_preds, all_probs, history

# --- 5. PLOTTING FUNCTIONS ---
def plot_training_curves(fold_histories):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for i, h in enumerate(fold_histories):
        plt.plot(h['loss'], alpha=0.4, label=f'Fold {i+1}')
    # Plot Mean
    avg_loss = np.mean([h['loss'] for h in fold_histories], axis=0)
    plt.plot(avg_loss, color='black', linewidth=2, label='Mean Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for i, h in enumerate(fold_histories):
        plt.plot(h['acc'], alpha=0.4, label=f'Fold {i+1}')
    avg_acc = np.mean([h['acc'] for h in fold_histories], axis=0)
    plt.plot(avg_acc, color='black', linewidth=2, label='Mean Acc')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Saved training_curves.png")

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Aggregate Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Saved confusion_matrix.png")

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    print("Saved roc_curve.png")

# --- MAIN ---
if __name__ == "__main__":
    X, y, ids = prepare_data()
    print(f"\nTotal Processed: {len(X)} slides")
    
    if len(X) < K_FOLDS:
        print("Not enough data for K-Fold.")
        exit()

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    
    fold_accuracies = []
    agg_labels = []
    agg_preds = []
    agg_probs = [] # Prob of Class 1
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        acc, lbls, preds, probs, history = train_fold(fold, train_idx, val_idx, X, y)
        fold_accuracies.append(acc)
        agg_labels.extend(lbls)
        agg_preds.extend(preds)
        agg_probs.extend(probs)
        fold_histories.append(history)
        
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.2%} ± {np.std(fold_accuracies):.2%}")
    
    class_names = ["High Steatosis", "Low Steatosis"]
    print(classification_report(agg_labels, agg_preds, target_names=class_names))
    
    # GENERATE PLOTS
    print("\nGenerating Plots...")
    plot_training_curves(fold_histories)
    plot_confusion_matrix(agg_labels, agg_preds, class_names)
    plot_roc_curve(agg_labels, agg_probs)
    print("All plots saved.")