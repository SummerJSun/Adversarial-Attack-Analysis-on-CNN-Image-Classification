# Adversarial Attack Analysis on Image Classification

A research project exploring the effectiveness of physical adversarial attack strategies on different models.

## Authors
- Lupin Cai (lupin.cai@emory.edu)
- Helen Jin (helen.jin@emory.edu)
- Jinghan Sun (jinghan.sun@emory.edu)

## Project Overview
This project investigates how different physical adversarial attack strategies affect CNN performance in image classification tasks. We focus on:

- Developing a CNN model for classifying images from the CIFAR-10 dataset
- Implementing and analyzing various adversarial patterns
- Evaluating the impact of these patterns on classification accuracy
- Testing against existing defense mechanisms

## Technical Requirements
- Python 3.11.5
- Libraries:
  - TensorFlow
  - PyTorch
  - Other machine learning and image processing libraries

## Dataset
We use the CIFAR-10 dataset, which includes:
- 60,000 color images (32x32 pixels)
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 50,000 training images and 10,000 test images
- 6,000 images per class

## Model Architecture
Our CNN model features:
- Three convolutional layers with batch normalization
- Max-pooling after each layer
- Adaptive average pooling
- Dropout layer for overfitting prevention
- Final fully connected layer with 10 class outputs

## Current Progress
- Successfully trained CNN model on CIFAR-10 dataset
- Achieved >80% test accuracy
- Implemented basic adversarial pattern generation
- Developed framework for testing different perturbation techniques

## Future Steps
Generate and analyze various dot patterns:
- Random distributions
- Clustered patterns
- Structured formations

Additional goals:
- Test different spatial configurations
- Evaluate impact of various noise patterns
- Compare performance against defense mechanisms

## References
1. K. Eykholt et al., "Robust Physical-World Attacks on Deep Learning Visual Classification"
2. Feng et al., "Robust and Generalized Physical Adversarial Attacks via Meta-GAN"
3. Chen et al., "ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN Object Detector"
4. Xiang et al., "PatchGuard: A Provably Robust Defense against Adversarial Patches"
5. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images"
