
# 🌸 Flower Recognition Using Deep Learning

A deep learning-based flower classification system that uses **Convolutional Neural Networks (CNNs)** and **transfer learning** to recognize five flower species. The project benchmarks four pretrained CNN architectures, achieving a **98.50% test accuracy with EfficientNetB1**, and includes a Python desktop GUI for image classification.

## 📌 Project Overview

This project explores the effectiveness of different CNN architectures for multi-class image classification. Four pretrained deep learning models were trained, fine-tuned, and evaluated to identify the most accurate architecture for flower recognition.

The best-performing model, **EfficientNetB1**, was selected for integration into an interactive desktop application that allows users to upload flower images and receive predictions with confidence scores.

### Key Features

- Classification of five flower species: **Daisy, Dandelion, Rose, Sunflower, and Tulip**
- Benchmarking of four pretrained CNN architectures
- Transfer learning and fine-tuning using ImageNet weights
- Image preprocessing and data augmentation
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Desktop GUI for image upload and real-time inference
- Visualization of predictions and confidence scores

## 🧠 Deep Learning Models

Four CNN architectures were evaluated:

| Model | Test Accuracy |
|---|---:|
| MobileNetV2 | 92.01% |
| ResNet50 | 87.62% |
| InceptionV3 | 91.55% |
| **EfficientNetB1** | **98.50%** |

**EfficientNetB1 achieved the highest test accuracy**, outperforming the other evaluated architectures.

### EfficientNetB1 Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **98.50%** |
| Validation Accuracy | 96.56% |
| Training Accuracy | 96.88% |
| Test Loss | 0.0521 |
| ROC-AUC | 0.9895 |

EfficientNetB1 demonstrated strong classification performance and generalization across the five flower categories, making it the selected model for the final application.

## 📊 Dataset

The project uses the **Flowers Recognition** dataset, containing images from five categories:

- 🌼 Daisy
- 🌾 Dandelion
- 🌹 Rose
- 🌻 Sunflower
- 🌷 Tulip

The dataset includes variations in lighting, image resolution, backgrounds, and flower orientation, providing a realistic image classification task.

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed class distributions and image dimensions.
- Visualized sample images from each category.
- Examined RGB color distributions.
- Investigated dataset variability.

### 2. Image Preprocessing
- Resized images to model-specific input dimensions.
- Applied architecture-specific preprocessing functions.
- Prepared training, validation, and testing datasets.

### 3. Data Augmentation
Applied multiple augmentation techniques to improve model generalization:

- Random rotations
- Horizontal and vertical shifts
- Random zooming
- Shearing
- Horizontal flipping

### 4. Model Training
- Initialized CNN architectures using pretrained ImageNet weights.
- Adapted classification layers for five flower categories.
- Applied transfer learning and fine-tuning.
- Monitored training and validation performance.

### 5. Model Evaluation
Evaluated each model using:

- Test accuracy and loss
- Precision, recall, and F1-score
- Classification reports
- Confusion matrices
- ROC curves and AUC scores
- Training and validation accuracy/loss curves

### 6. GUI Development
Developed a desktop graphical interface using **Python Tkinter** to demonstrate the final EfficientNetB1 classifier.

The application allows users to:
1. Select a flower image.
2. Preview the selected image.
3. Run classification using the trained model.
4. View the predicted flower category and confidence score.

## 🖥️ Application

The final application uses **EfficientNetB1** for image classification.

Example prediction:

| Output | Value |
|---|---|
| Predicted Class | Rose |
| Confidence | 1.00 |
| Actual Class | Rose |

The interface provides a straightforward way to demonstrate the trained model's inference capabilities.

## 🛠️ Technologies Used

**Programming Language**
- Python

**Deep Learning**
- TensorFlow
- Keras
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Fine-Tuning

**Data Processing & Visualization**
- NumPy
- Matplotlib
- Seaborn
- ImageDataGenerator

**Model Evaluation**
- Classification Metrics
- Confusion Matrices
- ROC-AUC Analysis

**Application Development**
- Tkinter
- Pillow (PIL)

## 📁 Repository Structure

```text
Flower-Recognition/
│
├── models/
│   └── EfficientNetB1/
│       └── class_indices.json
│
├── gui.py
├── project_code.ipynb
├── test.ipynb
├── .gitignore
└── README.md
```

### File Descriptions

- **project_code.ipynb** — Model development, training, evaluation, and comparison.
- **test.ipynb** — Model testing and prediction experiments.
- **gui.py** — Desktop application for flower classification.
- **models/EfficientNetB1/class_indices.json** — Mapping between flower categories and class indices.

## 🚀 Running the Application

Clone the repository:

```bash
git clone https://github.com/SondosGafar/Flower-Recognition.git
cd Flower-Recognition
```

Install the main GUI dependencies:

```bash
pip install tensorflow numpy pillow
```

The GUI requires the trained EfficientNetB1 model at:

```text
models/EfficientNetB1/model_224.keras
```

Once the trained model is available at that path, launch the application:

```bash
python gui.py
```

## 📈 Key Findings

- EfficientNetB1 achieved the highest test accuracy at **98.50%**.
- MobileNetV2 achieved **92.01%**, demonstrating a strong balance between accuracy and computational efficiency.
- InceptionV3 achieved **91.55%**, while ResNet50 achieved **87.62%**.
- Transfer learning and image augmentation supported effective multi-class flower classification.
- The final GUI demonstrates practical deployment of the selected model in a desktop application.

## 🔮 Future Improvements

- Expand the dataset to include additional flower species.
- Explore advanced fine-tuning and augmentation techniques.
- Optimize inference performance for resource-constrained devices.
- Develop a web-based interface for broader accessibility.

## 🎯 Conclusion

This project demonstrates an end-to-end deep learning workflow, including dataset exploration, image preprocessing, transfer learning, CNN benchmarking, model evaluation, and GUI development.

**EfficientNetB1 emerged as the best-performing architecture with 98.50% test accuracy**, highlighting the effectiveness of transfer learning for flower image classification.
