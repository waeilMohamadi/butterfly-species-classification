import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import pandas as pd
import os

from dataset import get_transforms

# Configuration
MODEL_PATH = 'butterfly_resnet50.pth'
TRAIN_CSV = 'Training_set.csv'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_class_names(csv_path):
    """Load unique class names from the training CSV."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        return sorted(df['label'].unique())
    return None

@st.cache_resource
def load_model(model_path, num_classes):
    """Load the trained ResNet50 model."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        return None
        
    model = model.to(DEVICE)
    model.eval()
    return model

def main():
    st.title("🦋 Butterfly Species Classifier")
    st.write("Upload an image of a butterfly to identify its species.")

    # 1. Load Data/Labels
    class_names = load_class_names(TRAIN_CSV)
    if class_names is None:
        st.error(f"Could not load labels from {TRAIN_CSV}. Make sure the file exists.")
        return

    # 2. Load Model
    model = load_model(MODEL_PATH, len(class_names))
    if model is None:
        st.error(f"Could not load model from {MODEL_PATH}. Please run training first.")
        return

    # 3. File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        st.write("Classifying...")
        
        # Preprocess
        transform = get_transforms(mode='val')
        img_tensor = transform(image).unsqueeze(0) # Add batch dimension
        img_tensor = img_tensor.to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
            
            pred_label = class_names[pred_idx.item()]
            conf_score = confidence.item() * 100

        # Output
        st.success(f"### Prediction: **{pred_label}**")
        st.info(f"Confidence: {conf_score:.2f}%")
        
        # Optional: Show Top 3
        st.write("---")
        st.write("#### Top 3 Probabilities:")
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        for i in range(3):
            label = class_names[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item() * 100
            st.write(f"{i+1}. **{label}**: {prob:.2f}%")

if __name__ == "__main__":
    main()
