# Adapting and Training VGG-11 for Traffic Sign Recognition

## Overview
This project adapts and trains the **VGG-11** convolutional neural network to classify images from the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. By leveraging transfer learning, the pretrained model on **ImageNet** is repurposed for traffic sign recognition, with a two-phase training process:
1. **Feature Extraction**: Freezes pretrained weights and trains only the classifier layers.
2. **Fine-Tuning**: Unfreezes all layers and fine-tunes the entire network for optimal performance.

---

## Features
- **Dataset**:
  - Uses the **GTSRB dataset**, which contains 43 classes of traffic signs.
  - Includes preprocessing, normalization, and dataset splitting for training, validation, and testing.

- **Model Architecture**:
  - **VGG-11** pretrained on ImageNet.
  - Modified classifier to match the number of GTSRB classes (43).

- **Training Process**:
  - Phase 1: Freezes pretrained weights to use the model as a feature extractor.
  - Phase 2: Unfreezes all layers for fine-tuning.

- **Performance Tracking**:
  - Tracks and visualizes training, validation, and test loss/accuracy.
  - Includes confusion matrix visualization to analyze model predictions.

---

## Key Components
### 1. **Dataset Preparation**
- **Data Augmentation and Preprocessing**:
  - Resizes all images to `224x224` to match VGG-11 input requirements.
  - Normalizes pixel values for consistent training.
- **Data Splits**:
  - Training: 75% of the dataset.
  - Validation: 25% of the training data.
  - Test: Independent test split.

### 2. **Model Adaptation**
- **Classifier Modification**:
  - Replaces the final fully connected layer of VGG-11 to output 43 logits (matching the number of traffic sign classes).
- **Weight Freezing and Unfreezing**:
  - Freezes pretrained layers during phase one.
  - Unfreezes all layers during phase two for comprehensive fine-tuning.

### 3. **Training Pipeline**
- **Phase One**:
  - Trains the classifier while keeping pretrained layers frozen.
- **Phase Two**:
  - Fine-tunes the entire model by unfreezing all weights.
- **Loss and Optimization**:
  - CrossEntropyLoss for multi-class classification.
  - Adam optimizer with dynamic learning rate adjustment.

### 4. **Evaluation and Visualization**
- Tracks metrics (loss and accuracy) across training, validation, and test phases.
- Visualizes training curves for both phases.
- Plots a confusion matrix to evaluate classification performance.

---

## Results
- **Performance Metrics**:
  - Achieves high accuracy on the test dataset, demonstrating the effectiveness of transfer learning and fine-tuning.
- **Visualization Outputs**:
  - Training curves showing loss and accuracy trends.
  - Confusion matrix highlighting class-specific performance.

---

### Prerequisites
- Python 3.x
- Libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`

## Future Enhancements
- Incorporate data augmentation techniques to improve generalization.
- Experiment with other pretrained models (e.g., ResNet, EfficientNet).
- Add a detailed error analysis to identify misclassified samples.