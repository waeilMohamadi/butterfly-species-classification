import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ButterflyDataset, get_transforms

# Configuration
CSV_FILE = 'Training_set.csv'
IMAGE_DIR = 'train'
BATCH_SIZE = 32
FAST_MODE = False # Enable for quick check

if FAST_MODE:
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.001
    DATA_SUBSET_SIZE = 200 # Only use 200 images for quick test
else:
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    DATA_SUBSET_SIZE = None

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader, desc=f"{phase} Phase"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    full_dataset = ButterflyDataset(csv_file=CSV_FILE, root_dir=IMAGE_DIR, transform=None)
    
    indices = range(len(full_dataset))
    if DATA_SUBSET_SIZE:
        print(f"FAST MODE: Using subset of {DATA_SUBSET_SIZE} images.")
        indices = indices[:DATA_SUBSET_SIZE]

    # Split indices (stratified split is better but random is okay for now)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=None # Stratify might fail with small random subset
    )
    
    # Create Subsets with appropriate transforms
    train_dataset = Subset(
        ButterflyDataset(csv_file=CSV_FILE, root_dir=IMAGE_DIR, transform=get_transforms('train')),
        train_indices
    )
    val_dataset = Subset(
        ButterflyDataset(csv_file=CSV_FILE, root_dir=IMAGE_DIR, transform=get_transforms('val')),
        val_indices
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility often
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # 2. Define Model (ResNet50 Transfer Learning)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all layers (optional: we can unfreeze some last layers, but let's start with feature extraction)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Train
    model, train_acc, val_acc, train_loss, val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS
    )
    
    # 4. Save Model
    torch.save(model.state_dict(), 'butterfly_resnet50.pth')
    print("Model saved to butterfly_resnet50.pth")

    # 5. Plot Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")

if __name__ == "__main__":
    main()
