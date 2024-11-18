# Pothole Detection Project

This project involves building a classification model to detect potholes in road images using a pre-trained model as a baseline and enhancing it with additional layers for improved performance.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The goal of this project is to classify road images into two categories: `Potholes` and `Normal`. 
- **Baseline Model**: A pre-trained ResNet18 model with the last layer replaced to match the dataset classes.
- **Enhanced Model**: Added dropout, batch normalization, and fully connected layers to improve accuracy and prevent overfitting.
- Metrics used: Accuracy, Precision, Recall, F1-Score.

---

## Features
- **Baseline Model**: Achieved basic classification with pre-trained ResNet18.
- **Enhanced Model**: Improved classification by adding layers and tuning hyperparameters.
- Visualized the results with sample predictions using Streamlit.

---

## Setup
### Prerequisites
- Python 3.7 or above
- Libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `streamlit`, `localtunnel`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aruzhankenzhebek/Pothole_Detection.git
   cd Pothole_Detection
