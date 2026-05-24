# TruthLens AI

TruthLens AI is a multimodal AI-generated content detection system designed to identify whether content is human-created or generated using modern AI models. The project combines a Computer Vision pipeline for image forensics with an NLP pipeline for AI-text detection, all integrated into a FastAPI-based web application.

The system contains two independent modules:

- **Image Forensics Module** using a custom CNN
- **Text Analysis Module** using fine-tuned DistilBERT

---

# Features

- AI vs Human Image Detection
- AI vs Human Text Detection
- FastAPI backend for real-time inference
- Confidence score predictions
- CNN-based image forensic analysis
- DistilBERT-based text classification
- Interactive web interface
- Separate training notebooks included

---

# Project Structure

```bash
TruthLens-AI/
│
├── main.py
├── fix_model.py
│
├── models/
│   ├── ai_text_detector/
│   └── ai_vs_human_cnn.h5
│
├── static/
│   └── index.html
│
├── Model Training Scripts/
│   ├── CNN_TruthLens.ipynb
│   └── Fine_Tuning(Text_Module).ipynb
│
└── README.md
```

---

# Modules Overview

## 1. Image Forensics Module

The image detection pipeline uses a custom Convolutional Neural Network (CNN) trained to classify images as either:

- Human/Real
- AI Generated

### Image Preprocessing

- Resize to `224x224`
- Normalization to `[0,1]`
- Data augmentation:
  - Horizontal flipping
  - Rotation
  - Zoom
  - Width/Height shifting

### CNN Architecture

- 3 Convolutional Blocks
  - 32 Filters
  - 64 Filters
  - 128 Filters
- Batch Normalization
- MaxPooling
- Dense Layers
- Dropout Regularization
- Sigmoid Output Layer

### Performance

| Metric | Score |
|---|---|
| Accuracy | 88% |
| Human Recall | 0.94 |
| AI Precision | 0.93 |

---

## 2. Text Analysis Module

The text detection module uses a fine-tuned `distilbert-base-uncased` model for binary classification of:

- Human Written Text
- AI Generated Text

### Text Preprocessing

- Lowercasing
- URL removal
- Special character cleaning
- Whitespace normalization
- Tokenization using DistilBERT tokenizer
- Maximum sequence length: `128`

### Fine-Tuning Features

- AdamW Optimizer
- Learning Rate Warmup + Decay
- FP16 Mixed Precision Training
- Gradient Accumulation
- Label Smoothing
- Early Stopping

### Performance

| Metric | Score |
|---|---|
| Accuracy | ~97% |
| Human F1 | 0.974 |
| AI F1 | 0.963 |

---

# Dataset Information

## Image Dataset

Dataset Used:
- Kaggle — **AI vs Human Generated Dataset**
- Author: `alessandrasala79`

Total Images:
- 12,000 balanced samples

Split:
- 80% Training
- 20% Validation

---

## Text Dataset

A custom combined dataset (`combined_cleaned.csv`) was used.

Dataset preparation included:
- Removing texts under 10 words
- Balancing classes
- Length bucketing
- Adding casual human-written samples for generalization

Split:
- 80% Training
- 10% Validation
- 10% Testing

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-link>
cd TruthLens-AI
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Download Model Files

Due to GitHub file size limitations, trained model files are not included in this repository.

Download the following model files manually and place them inside the `models/` folder.

## Required Files

### CNN Model
Place:
```bash
models/ai_vs_human_cnn.h5
```

Download:
```text
https://drive.google.com/file/d/1U-6NNjyl3RuSQYQU2wwmhNBA5msatFLu/view?usp=sharing
```

---

### DistilBERT Text Model Folder
Place:
```bash
models/ai_text_detector/
```

Download:
```text
https://drive.google.com/drive/folders/16Sa7PzxuAdQOgTph6Q6y-F9OPMEUtzGy?usp=sharing
```

---

# Running the Application

Start the FastAPI server:

```bash
python main.py
```

The application will run locally and can be accessed through the browser.

---

# Technologies Used

## Backend
- Python
- FastAPI

## Deep Learning
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers

## Frontend
- HTML
- CSS
- JavaScript

## Other Libraries
- NumPy
- OpenCV
- Scikit-learn
- Pandas

---

# Training Scripts

The repository contains separate notebooks for training both modules.

| Notebook | Purpose |
|---|---|
| `CNN_TruthLens.ipynb` | CNN Image Detector Training |
| `Fine_Tuning(Text_Module).ipynb` | DistilBERT Fine-Tuning |

---

# Current Limitations

- Image CNN has lower recall for AI-generated images
- No evaluation on newer generators like:
  - Midjourney v6
  - Stable Diffusion 3
  - DALL·E 3
- Short text detection remains challenging
- Text endpoint is not fully deployed in production
- No adversarial robustness testing
- No multimodal fusion implemented yet

---

# Future Improvements

- Cross-modal fusion
- Explainable AI visualizations
- Better generalization across generators
- Adversarial robustness testing
- Larger and more diverse datasets
- Production deployment for both modules

---

# Research Motivation

With the rapid advancement of generative AI models, distinguishing AI-generated content from human-created content is becoming increasingly difficult. TruthLens AI aims to address this challenge by combining Computer Vision and NLP techniques to improve digital content authenticity verification.

---

# Author

Developed as a deep learning project focused on AI-generated content detection using multimodal AI techniques involving Computer Vision and Natural Language Processing.
