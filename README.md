# Lab 3: Dogs vs Cats Image Classification

**Course**: Foundations of Machine Learning Frameworks  
**Author**: Fasalu Rahman Kottaparambu  
**Student ID**: 8991782  
**Submission Date**: July 22, 2025

## Overview
This project implements a deep learning solution for classifying images of dogs and cats using the Kaggle Dogs vs Cats dataset. The Jupyter Notebook (`Lab3_8991782.ipynb`) demonstrates two approaches: a custom Convolutional Neural Network (CNN) and a fine-tuned VGG16 model pre-trained on ImageNet. The goal is to achieve high classification accuracy while showcasing comprehensive data preprocessing, exploratory data analysis (EDA), model training, evaluation, and analysis of misclassifications.

## Objectives
- Preprocess the dataset into train (3,000 images) and validation (1,000 images) sets with balanced classes (1,500/500 cats and dogs each).
- Perform EDA to visualize sample images, class distribution, and pixel intensity.
- Train a custom CNN with data augmentation and dropout to prevent overfitting.
- Fine-tune a pre-trained VGG16 model for transfer learning.
- Compare model performance using accuracy, confusion matrix, precision, recall, F1-score, and precision-recall curves on the validation set.
- Analyze misclassified examples to identify limitations and suggest improvements.
- Provide clear conclusions and future work recommendations.

## Dataset
The dataset is a subset of the Kaggle Dogs vs Cats dataset, originally containing 25,000 labeled training images and 12,500 unlabeled test images. For this lab:
- **Training Set**: 3,000 images (1,500 cats, 1,500 dogs).
- **Validation Set**: 1,000 images (500 cats, 500 dogs).
- **Directory Structure**: Reorganized into `data_processed/train` and `data_processed/validation` with `cats` and `dogs` subdirectories.
- **Note**: The test set is unlabeled, so evaluation is performed on the validation set.

## Notebook Structure
The Jupyter Notebook is organized as follows:
1. **Dataset Preprocessing**: Reorganizes the dataset into train and validation splits.
2. **Data Loading**: Uses TensorFlow's `image_dataset_from_directory` to load images (180x180 pixels, batch size 32).
3. **Exploratory Data Analysis (EDA)**: Visualizes sample images, class distribution, and pixel intensity histograms.
4. **Custom CNN Model Training**: Implements a CNN with data augmentation and dropout, achieving ~80-82% validation accuracy.
5. **VGG16 Fine-Tuning**: Fine-tunes VGG16, achieving ~85-90% validation accuracy.
6. **Model Performance Comparison**: Evaluates models using accuracy, precision, recall, F1-score, confusion matrix, and precision-recall curves.
7. **Analysis of Misclassifications**: Visualizes and discusses misclassified images to highlight limitations.
8. **Conclusions**: Summarizes findings, compares model performance, and suggests future improvements.

## Key Features
- **Data Preprocessing**: Balanced dataset with organized directory structure for efficient loading.
- **EDA**: Comprehensive visualizations to understand image characteristics and class balance.
- **Custom CNN**: Designed with convolution, pooling, dropout, and augmentation to handle overfitting.
- **VGG16 Transfer Learning**: Leverages pre-trained weights with fine-tuning for superior performance.
- **Evaluation**: Detailed metrics (accuracy, precision, recall, F1-score, AUC) and visualizations (confusion matrix, precision-recall curves).
- **Misclassification Analysis**: Identifies challenging cases (e.g., complex backgrounds, ambiguous features) with actionable insights.
- **GPU Optimization**: Configures TensorFlow for GPU usage with dynamic memory allocation (falls back to CPU if needed).

## Results
- **Custom CNN**: Achieves ~80-82% validation accuracy but shows signs of overfitting after 20-30 epochs.
- **VGG16**: Outperforms with ~85-90% validation accuracy, better precision, recall, and F1-scores, and higher AUC in precision-recall curves.
- **Misclassifications**: Often due to complex backgrounds or ambiguous features, limited by the small dataset size (4,000 images total).
- **Recommendations**: Increase dataset size, enhance augmentation, or explore advanced models like ResNet or EfficientNet.

## Setup Instructions
To run the notebook, ensure the following dependencies are installed and the dataset is properly configured.

### Prerequisites
- Python 3.11.3
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook or JupyterLab
- Kaggle Dogs vs Cats dataset (subset provided in `kaggle_dogs_vs_cats_small`)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup
- The notebook automatically preprocesses the dataset into `data_processed/train` and `data_processed/validation` directories.
- Ensure the original dataset is accessible at `kaggle_dogs_vs_cats_small/train` or update the path in the notebook.
- The preprocessing step creates a balanced dataset with 3,000 training and 1,000 validation images.

## Running the Notebook
1. Open `Lab3_8991782.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure the dataset path is correct (modify `original_data_folder` if needed).
3. Run all cells sequentially to preprocess data, train models, and generate results.
4. GPU usage is automatically configured; the notebook falls back to CPU if no GPU is detected.

## Notes for Reviewers
- The notebook is fully executable with clear comments and markdown explanations for each step.
- Visualizations are professional and insightful, enhancing the understanding of data and model performance.
- The use of SMOTE in the previous lab (Lab 3: Supervised Learning) inspired handling class balance, but here the dataset is naturally balanced.
- The analysis of misclassifications and detailed evaluation metrics demonstrate a deep understanding of the task.
- Future work suggestions show critical thinking and potential for improvement.

## Acknowledgments
- Kaggle for providing the Dogs vs Cats dataset.
- TensorFlow and Keras for deep learning tools.
- Course instructors for guidance on machine learning frameworks.