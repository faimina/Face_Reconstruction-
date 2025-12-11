# Facial Recognition System for Identity Verification & Identification

This project implements multiple deep-learning architectures (Autoencoder, Siamese Network, Triplet Network, and MobileNetV2 feature extraction) for facial identity verification and identification using the Labeled Faces in the Wild (LFW) dataset. The system performs face reconstruction, embedding generation, similarity measurement, and identity classification.

---------------------------------------------------------------------

## 📌 Project Overview

This project solves two main tasks:

### 1. Identity Verification
Determining if two face images belong to the same person.
Models:
- Custom CNN Siamese Network
- VGG16 Siamese Network
- Fine-tuned VGG16 Siamese
- MobileNetV2 embedding + threshold verification

### 2. Identity Identification
Predicting the exact identity.
Models:
- MobileNetV2 frozen feature extractor
- Dense classifier head

The project also includes a deep Autoencoder for reconstruction and unsupervised feature learning.

---------------------------------------------------------------------

## 📂 Dataset (LFW) Overview + Samples

The LFW dataset contains 13,000+ labeled faces of 5,749 individuals.

Dataset includes:
- people.csv — list of names + number of images
- pairs.csv — pairs of images labeled same/different person

### Sample from people.csv

name,n_images
George_W_Bush,530
Colin_Powell,236
Tony_Blair,144
Ariel_Sharon,77
Donald_Rumsfeld,121

### Sample from pairs.csv

same,person1,img1,person2,img2
1,George_W_Bush,0001,George_W_Bush,0002
1,Colin_Powell,0003,Colin_Powell,0004
0,Tony_Blair,0001,George_W_Bush,0005
0,Donald_Rumsfeld,0002,Ariel_Sharon,0001

---------------------------------------------------------------------

## 🧠 Models Implemented

### 1. Autoencoder (Unsupervised Facial Embeddings)
- Learns 128-dimensional embedding
- Loss: MSE + SSIM
- Used for reconstruction + latent space visualization (t-SNE)

### 2. MobileNetV2 (Frozen Features)
- Backbone = frozen
- Outputs 1280-dim feature vector
- Only classifier head trained
- Validation accuracy ≈ 15% (expected due to limited data subset)

### 3. Siamese Networks (Face Verification)

| Model Version                    | Validation Accuracy |
|----------------------------------|---------------------|
| Custom CNN Siamese               | 62.75%              |
| VGG16 Siamese (Frozen + Aug)     | 61.75%              |
| Fine-tuned VGG16 Siamese         | 67.25%              |

Loss: Contrastive (L1 distance between embeddings)

### 4. Triplet Network (MobileNetV2 Backbone)
- Triplet Loss (Anchor, Positive, Negative)
- Produces 128-dim L2-normalized embeddings
- Validation accuracy: 78.75%

---------------------------------------------------------------------

## 📊 Model Performance Summary

- Autoencoder → high-quality reconstructions + stable latent space
- Siamese fine-tuned VGG16 → 67% verification accuracy
- Triplet Network → 78% verification accuracy
- MobileNetV2 frozen classifier → weak generalization (expected without fine-tuning)

---------------------------------------------------------------------

## 📁 Project Structure

/autoencoder/
train_autoencoder.py
autoencoder_model.h5
reconstruction_results/

/siamese/
siamese_custom.py
siamese_vgg16.py
siamese_finetuned.py
checkpoints/

/triplet/
triplet_model.py
triplet_utils.py
embeddings/

/pretrained_model/
mobilenet_features.py
classifier_head.py

/data/
people.csv
pairs.csv
lfw_images/

/results/
accuracy_plots/
loss_curves/
confusion_matrices/

---------------------------------------------------------------------

## ▶️ How to Run

### Autoencoder

python autoencoder/train_autoencoder.py

### Siamese Networks

python siamese/siamese_custom.py
python siamese/siamese_vgg16.py
python siamese/siamese_finetuned.py

### Triplet Network

python triplet/triplet_model.py

### MobileNetV2 Frozen Classifier

python pretrained_model/mobilenet_features.py

---------------------------------------------------------------------

## 🔧 Requirements (requirements.txt)

tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
opencv-python
scipy

---------------------------------------------------------------------

## 👥 Contributors
- Faimina Khokhani
- Nate Sternberg
- Jorge Martinez
- Sergio Medrano
- Sepideh Forouzi

---------------------------------------------------------------------

## 📄 License
Academic use only. Not licensed for commercial deployment.

