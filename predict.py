import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from tqdm import tqdm

from dataset import ButterflyDataset, get_transforms

# Configuration
TEST_CSV = 'Testing_set.csv'
TEST_DIR = 'test'
TRAIN_CSV = 'Training_set.csv' # Needed for label map
MODEL_PATH = 'butterfly_resnet50.pth'
OUTPUT_FILE = 'submission.csv'
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Label Map from Training Set
    # We need to know the mapping from ID to Label to decode predictions
    train_dataset = ButterflyDataset(csv_file=TRAIN_CSV, root_dir='train') # root_dir doesn't matter here
    if train_dataset.id_to_label is None:
        print("Error: Could not determing label mapping from training set.")
        return
    id_to_label = train_dataset.id_to_label
    num_classes = len(train_dataset.classes)
    
    # 2. Prepare Test Data
    test_dataset = ButterflyDataset(
        csv_file=TEST_CSV, 
        root_dir=TEST_DIR, 
        transform=get_transforms('val'),
        label_map=None # No labels in test set
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    model = models.resnet50(weights=None) # No need to download weights again, we will load ours
    # Recreate structure
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded model weights.")
    else:
        print(f"Error: Model file {MODEL_PATH} not found. Train the model first.")
        return
        
    model = model.to(DEVICE)
    model.eval()
    
    # 4. Inference
    predictions = []
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(filenames)):
                filename = filenames[i]
                pred_idx = preds[i].item()
                pred_label = id_to_label[pred_idx]
                
                predictions.append({'filename': filename, 'label': pred_label})
                
    # 5. Save Results
    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
