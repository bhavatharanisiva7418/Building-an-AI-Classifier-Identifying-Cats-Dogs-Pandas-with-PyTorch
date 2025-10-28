# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch

## AIM:
Build an image classifier that can identify whether an image contains a cat, dog, or panda using transfer learning (ResNet18) in PyTorch.

## Procedure:
## 1. Environment Setup
Ensure Python 3.9 or higher is installed.

Install PyTorch and torchvision libraries.

Verify CUDA (GPU) availability to speed up training.

If using Kaggle, enable GPU accelerator from the notebook settings.

## 2. Dataset Preparation
Download the Cats vs Dogs vs Pandas dataset from Kaggle.

Organize the dataset into separate folders for training and testing:

Each with subfolders named cat, dog, and panda.

Make sure each folder contains only valid image files (e.g., JPG or PNG).

Remove any hidden folders (like .ipynb_checkpoints) that might cause errors.

## 3. Data Loading and Transformation
Use PyTorch’s ImageFolder to load images from the directories.

Apply data transformations:

Resize all images to 224x224 pixels.

For training data, apply data augmentation such as random horizontal flips.

Normalize images using the standard ImageNet mean and standard deviation.

Apply only resizing and normalization to test data.

## 4. Model Design Using Transfer Learning
Load a pretrained ResNet18 model from torchvision.

Freeze all convolutional layers to retain pretrained weights.

Replace the final fully connected layer with a new classifier suitable for 3 classes (cat, dog, panda).

Move the model to GPU if available.

## 5. Training the Model
Set a random seed for reproducibility.

Use CrossEntropyLoss as the loss function.

Use Adam optimizer with a learning rate of 0.001.

Train the model for 10 to 15 epochs.

Save the best model checkpoint based on validation accuracy.

## 6. Model Evaluation
Evaluate the trained model on the test dataset.

Calculate test loss and accuracy.

Plot a confusion matrix to visualize performance across classes.

Display example images alongside their predicted and true labels.
