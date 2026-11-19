import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ButterflyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_map=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            label_map (dict, optional): Mapping from string labels to integers.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        
        # If labels are present, create a label map if not provided
        if 'label' in self.data_frame.columns:
            if self.label_map is None:
                self.classes = sorted(self.data_frame['label'].unique())
                self.label_map = {label: i for i, label in enumerate(self.classes)}
            self.id_to_label = {v: k for k, v in self.label_map.items()}
        else:
            self.classes = None
            self.id_to_label = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if isinstance(idx, float): idx = int(idx)
        
        row = self.data_frame.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or raise error? For now, let's error out to be safe.
            raise e

        if self.transform:
            image = self.transform(image)

        if 'label' in row:
            label_str = row['label']
            label = self.label_map[label_str]
            return image, label
        else:
            # For test set, return the image and the filename (ID)
            return image, img_name

def get_transforms(mode='train'):
    """
    Returns the data transforms for training or validation/testing.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

if __name__ == "__main__":
    # Simple test
    import sys
    
    # Paths (adjust as needed for local testing outside of this flow)
    # csv_path = "Training_set.csv"
    # img_dir = "train"
    
    pass
