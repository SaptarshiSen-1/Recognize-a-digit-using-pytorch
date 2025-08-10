# MNIST Handwritten Digit Recognition with PyTorch
This project implements a Convolutional Neural Network (CNN) in PyTorch to recognize handwritten digits from the MNIST dataset.
It also includes functionality to upload custom digit images in Google Colab and predict their labels.

# Features
CNN architecture optimized for MNIST digit classification
GPU support (automatically detects CUDA availability)
Image preprocessing:
Converts to grayscale
Inverts colors if needed
Resizes to 28×28 pixels
Normalizes to match MNIST training data
Colab file upload support for testing custom handwritten digit images

# Requirements
Install dependencies with:
bash
pip install torch torchvision matplotlib pillow
 Dataset
This project uses the MNIST dataset, automatically downloaded by torchvision.datasets.MNIST.

# It contains:
60,000 training images
10,000 testing images
Images are grayscale, 28×28 pixels, labeled 0–9
Model Architecture
The CNN consists of:
Conv2d (1 → 32), kernel size 3, ReLU
Conv2d (32 → 64), kernel size 3, ReLU
MaxPool2d (2×2)
Flatten layer
Linear (9216 → 128), ReLU
Dropout (0.25)
Linear (128 → 10)

# Training
By default, the model trains for 8 epochs with the Adam optimizer and a learning rate of 0.001.
To train:
python
 Run the training loop
for epoch in range(8):
    
# Testing
After each epoch, the test accuracy is calculated using the MNIST test set.
Example output:
python-repl
Epoch 1 Test Acc: 97.85%
Epoch 2 Test Acc: 98.42%
...

# Predicting Custom Images
Run the notebook in Google Colab.
Upload a custom digit image (.png, .jpg, etc.).
The model will preprocess and predict the digit.


Typical test accuracy: 98–99% after 8 epochs.
Accuracy may vary depending on preprocessing and training parameters.

 License
This project is open-source and available under the MIT License.
