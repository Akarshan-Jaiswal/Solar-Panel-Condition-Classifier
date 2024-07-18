# Solar Panel Condition Classifier

## Overview
This project is aimed at developing a model to recognize the state of solar panels and classify them into one of the following categories: 'Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', and 'Snow-Covered'. The model uses the VGG16 architecture as a base for feature extraction and is fine-tuned to achieve accurate classification results.

## Dataset
The dataset is organized into directories with subdirectories for each class. The images are automatically split into training and validation sets using TensorFlow's `image_dataset_from_directory` utility.

## Model Architecture
The model is based on the VGG16 architecture, pretrained on the ImageNet dataset. The top layers are removed, and a global average pooling layer, dropout layer, and dense layer are added for classification.

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- tensorflow
- opencv-python (cv2)
- glob

Install the required packages using pip:
```bash
pip install pandas numpy seaborn matplotlib tensorflow opencv-python glob
