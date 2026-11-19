import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Configuration
CSV_FILE = 'Training_set.csv'
IMAGE_DIR = 'train'

def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # 1. Basic Info
    print("\n--- Dataset Info ---")
    print(f"Total Images: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'label' not in df.columns:
        print("No 'label' column found. Skipping class analysis.")
        return

    # 2. Class Distribution
    print("\n--- Class Distribution ---")
    class_counts = df['label'].value_counts()
    print(f"Number of Classes: {len(class_counts)}")
    print(f"Most Common: {class_counts.idxmax()} ({class_counts.max()} images)")
    print(f"Least Common: {class_counts.idxmin()} ({class_counts.min()} images)")
    
    # Plot Class Distribution
    plt.figure(figsize=(20, 8))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Distribution of Butterfly Classes")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('eda_class_distribution.png')
    print("Saved plot to 'eda_class_distribution.png'")
    
    # 3. Sample Images
    print("\n--- Saving Sample Grid ---")
    unique_classes = df['label'].unique()[:9] # Take first 9 classes
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, cls in enumerate(unique_classes):
        row = i // 3
        col = i % 3
        
        sample_row = df[df['label'] == cls].iloc[0]
        img_path = os.path.join(IMAGE_DIR, sample_row['filename'])
        
        try:
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(cls)
            axes[row, col].axis('off')
        except Exception as e:
            print(f"Could not load {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig('eda_sample_images.png')
    print("Saved sample grid to 'eda_sample_images.png'")

if __name__ == "__main__":
    main()
