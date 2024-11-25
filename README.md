# Plant Disease Classification with PiT (Pooling-based Vision Transformer)
This repository contains the implementation of a deep learning pipeline for plant disease classification using the PiT model, along with support for ViT, and ResNet architectures.To evaluate the effectiveness of Pooling-based Vision Transformer (PiT) architecture for plant disease detection using the Plant Village dataset, and to conduct a comprehensive comparative analysis against traditional Vision Transformer (ViT) models in the context of agricultural applications. 

# Project Overview
The goal of this project is to explore the PiT architecture for plant disease detection using the Plant Village dataset. The results are compared against ViT, and ResNet focusing on their applicability to agricultural diagnostics.

# Table of Contents
1. Dataset Preparation
2. Preparation of the Dataset
3. Training the Model
4. Evaluating the Model
5. Model Performance Metrics
6. Visualization
7. Model Statistics
8. Authors
9. Disclaimer

# Dataset Preparation
# 1: Dataset Structure
Use the plant village dataset. So in Kaggle, go to inputs -> dataset -> and type in "plant village dataset", it should be the first one you see.

path = /kaggle/input/plant-village-dataset-updated

# Reorganizing and Renaming
Running the following function will reorganize and rename the dataset for better usability.

Function: def rename_and_merge(source_path, destination_path):

Classes: ['Apple_Apple_Scab', 'Apple_Black_Rot', 'Apple_Cedar_Apple_Rust', 'Apple_Healthy', 'Bell Pepper_Bacterial_Spot', 'Bell Pepper_Healthy', 'Cherry_Healthy', 'Cherry_Powdery_Mildew', 'Corn (Maize)_Cercospora_Leaf_Spot', 'Corn (Maize)_Common_Rust_', 'Corn (Maize)_Healthy', 'Corn (Maize)_Northern_Leaf_Blight', 'Grape_Black_Rot', 'Grape_Esca_(Black_Measles)', 'Grape_Healthy', 'Grape_Leaf_Blight', 'Peach_Bacterial_Spot', 'Peach_Healthy', 'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight', 'Strawberry_Healthy', 'Strawberry_Leaf_Scorch', 'Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Healthy', 'Tomato_Late_Blight', 'Tomato_Septoria_Leaf_Spot', 'Tomato_Yellow_Leaf_Curl_Virus']
Number of classes: 29
Training samples: 53690
Validation samples: 12067
Test samples: 1354

# 2: Create/Load in the Model
Loading in the using the script below to prepare it for training and testing.
import timm
import torch
from torch import nn

# Load the PiT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('pit_s_224', pretrained=True, num_classes=len(train_dataset.classes))
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

print("Model loaded successfully!")

# 3: Training the Model
  a. Ensure the reorganized dataset is available in your directory (see Dataset Preparation).
  b. Adjust training parameters in the script if needed (e.g., epochs, batch_size).
  c. Run the training script.

  d. Key Details
      Pre-trained PiT Model: Loaded from the timm library.
      Optimizer: AdamW with a learning rate of 0.001.
      Batch Size: 32.
      Epochs: 10 (can be adjusted).
      Training time: ~2.5-3.5 hours (on Kaggle with GPU P100).

  Function: def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10):

# 4: Evaluating the Model
  a. Classification Report: Precision, Recall, F1-Score for all 29 classes.
  
  b. Confusion Matrix: Visualized with seaborn.
  
  c. Predictions Visualization: Correctly and incorrectly classified images.

Function: def evaluate_model(model, test_loader):

# 5: Model Performance Metrics
  a. Training Accuracy: Recorded after each epoch.
  
  b. Validation Accuracy: Evaluated on the validation set.
  
  c. Testing Metrics: Overall performance on unseen test data.
  
  d. Key outputs include FLOPs, parameter count, and inference speed:
      GFLOPs: FlopCountAnalysis used for calculation.
      Inference Speed: Average processing time for GPU and CPU.

# 6: Visualization
  a. Visualize the smaples with correct (with confidence scores) and incorrect predictions (misclassified samples).
  
  b. Utilize a confusion matrix for the entire test set

# 7: Model Statistics
  1. FLOPs and Parameters
     a. FLOPs: Measured in GFLOPs using ptflops and FlopCountAnalysis.
     b. Parameters Count: Total number of trainable parameters.
  
  2. Inference Speed
     a. GPU Batch Time: Measured in milliseconds per batch and per image.
     b. CPU Batch Time: Averaged over 10 iterations for reliability.

# 8: Authors
This project was created by a group of 4 AUS students: -
   a. Siva Adduri
   
   b. Mohamed Alkhaja Alawadhi
   
   c. Arya Sankhe
   
   d. Hamza Khan 

# 9: Disclaimer: -
1. Software/Interface utilized was Kaggle to code the project.
2. You should first optimize your "Session options" in Kaggle.
   a. Turn on the internet (you would first have to "Get phone verified" to use the internet).
   b. Use GPU instead of CPU to compile the codes (Accelerator -> either "GPU T4 x2" or "GPU P100", in this project we used "GPU P100").
3. The train_and_validate code will compile the code fully on average of 2.5 to 3.5 hours (for each model).
4. While each model (PiT, ViT, ResNet, EfficientNet) shares a similar structure, some variations exist to accommodate model-specific requirements.
5. This final project was required for a Computer Vision Course at the American University of Sharjah (AUS).
