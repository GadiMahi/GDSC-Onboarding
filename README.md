# CIFAR-10 Image Classification using Convolutional Neural Networks and Transfer Learning

## Project Overview
This project aims to classify images from the **CIFAR-10 dataset** using **deep learning techniques**. The model progression follows a structured approach, improving accuracy at each stage by incorporating **regularization, data augmentation, hyperparameter tuning, and transfer learning**. The final model leverages **ResNet50** for optimal performance.

---

## Project Workflow and Model Progression
The project was developed in multiple stages, with systematic improvements.

### Stage 1: Initial CNN Model (73% Accuracy)
- Implemented a **basic CNN model** with three convolutional layers.
- Used **ReLU activation** and **softmax output for classification**.
- Optimized using **Adam optimizer** with a learning rate of **0.001**.
- Regularization applied using **Dropout (0.3)**.
- Achieved **73% test accuracy**.

### Stage 2: Optimized CNN Model (85.16% Accuracy)
- Integrated **Batch Normalization** to stabilize training.
- Increased regularization with **Dropout (0.5)** to prevent overfitting.
- Added **L1/L2 weight regularization**.
- Implemented **learning rate scheduling** using `ReduceLROnPlateau`.
- Improved generalization with **data augmentation**.
- Achieved **85.16% test accuracy**.

### Stage 3: Transfer Learning with ResNet50 (95.87% Accuracy)
- Integrated **ResNet50 as a feature extractor** with fine-tuning enabled.
- Replaced the final layers with **custom dense layers** optimized for CIFAR-10.
- Applied **Global Average Pooling** instead of Flatten for better efficiency.
- Lowered the learning rate to **0.00005** to stabilize training.
- Applied **data augmentation** for enhanced generalization.
- Achieved **95.44% test accuracy**.

---

## Dataset Information
- **Dataset**: CIFAR-10
- **Number of Classes**: 10
- **Training Samples**: 50,000
- **Testing Samples**: 10,000
- **Image Dimensions**: 32x32 pixels with three color channels (RGB)

The dataset is available through `tensorflow.keras.datasets.cifar10`.

---

## Model Architecture

### ResNet50-Based Transfer Learning Model
| Layer | Details |
|--------|---------|
| Input | Image resizing to 224x224 |
| Data Augmentation | Random flip, rotation, and zoom |
| Pre-trained Model | ResNet50 (Pre-trained on ImageNet) |
| Feature Extractor | Fine-tuned last 10 layers |
| Pooling | Global Average Pooling |
| Fully Connected Layer | Dense (1024 neurons, ReLU) |
| Batch Normalization | Applied after Dense layers |
| Dropout | 50% |
| Output | Dense (10 classes, Softmax) |

---

## Implementation Details

### Dependencies
Ensure the following libraries are installed:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn
