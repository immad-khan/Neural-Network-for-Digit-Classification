# 🧠 Neural Network for Handwritten Digit Classification (MNIST)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> A comparative study of ReLU vs Sigmoid activation functions in feedforward neural 
> networks for multi-class image classification on the MNIST dataset.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution Approach](#-solution-approach)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Neural Network Architecture](#-neural-network-architecture)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project builds and trains **two feedforward neural networks** to classify 
handwritten digits (0-9) from the famous **MNIST dataset**. The primary focus is 
a **comparative analysis** between two popular activation functions:

| Model | Activation Function | Test Accuracy |
|-------|-------------------|---------------|
| **Model A** | ReLU (Rectified Linear Unit) | **~97-98%** |
| **Model B** | Sigmoid | **~96-97%** |

Both models share an identical architecture (3 hidden layers with 256, 128, and 
64 neurons), differing only in their hidden layer activation functions.

---

## ❓ Problem Statement

Handwritten digit recognition is a fundamental challenge in computer vision with 
real-world applications in:

- 📬 **Postal Services** — Automated reading of handwritten zip codes
- 🏦 **Banking** — Check amount verification
- 📄 **Document Digitization** — Converting handwritten forms to digital text
- 🎓 **Education** — Automated grading systems

**The Challenge:** Every person writes differently. A "7" written by one person 
looks completely different from another's. How do we teach a computer to handle 
this enormous variability?

---

## 💡 Solution Approach
Raw Images (28×28 pixels)
│
▼
┌─────────────────────┐
│ PREPROCESSING │
│ • Normalize (0-1) │
│ • Flatten (784) │
│ • One-Hot Encode │
└─────────┬───────────┘
│
┌─────┴─────┐
▼ ▼
┌────────┐ ┌────────┐
│Model A │ │Model B │
│ (ReLU) │ │(Sigmoid│
│ │ │ │
│ 784 │ │ 784 │
│ ↓ │ │ ↓ │
│ 256 │ │ 256 │
│ ↓ │ │ ↓ │
│ 128 │ │ 128 │
│ ↓ │ │ ↓ │
│ 64 │ │ 64 │
│ ↓ │ │ ↓ │
│ 10 │ │ 10 │
│(softmax│ │(softmax│
└───┬────┘ └───┬────┘
│ │
▼ ▼
~98% ~97%
accuracy accuracy


---

## 📊 Dataset

The **MNIST** (Modified National Institute of Standards and Technology) dataset:

| Property | Details |
|----------|---------|
| **Total Images** | 70,000 grayscale images |
| **Training Set** | 60,000 images |
| **Test Set** | 10,000 images |
| **Image Size** | 28 × 28 pixels |
| **Classes** | 10 (digits 0-9) |
| **Pixel Values** | 0 (black) to 255 (white) |

![Sample Digits](sample_digits.png)
*Sample images from the MNIST dataset showing the variety of handwriting styles.*

---

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/mnist-digit-classification-neural-network.git
cd mnist-digit-classification-neural-network
Step 2: Create a Virtual Environment
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Step 3: Install Dependencies
pip install -r requirements.txt
Expected Output
The script will:

✅ Load and preprocess the MNIST dataset
✅ Build two neural network models (ReLU and Sigmoid)
✅ Train both models for 20 epochs
✅ Display training progress in the terminal
✅ Generate and save 5 visualization plots
✅ Print a detailed performance comparison table
✅ Save both trained models
Estimated runtime: 2-5 minutes (depending on hardware)

🏗️ Neural Network Architecture
Both models share this identical architecture:
Layer (type)              Output Shape         Parameters
================================================================
hidden_layer_1 (Dense)    (None, 256)          200,960
hidden_layer_2 (Dense)    (None, 128)          32,896
hidden_layer_3 (Dense)    (None, 64)           8,256
output_layer (Dense)      (None, 10)           650
================================================================
Total params: 242,762
Trainable params: 242,762
Non-trainable params: 0
Training Configuration
Parameter	Value
Optimizer	Adam
Loss Function	Categorical Cross-Entropy
Epochs	20
Batch Size	128
Validation Split	20%
📈 Results
Final Test Set Performance
Metric	Model A (ReLU)	Model B (Sigmoid)	Winner
Test Accuracy	~97.85%	~96.92%	✅ ReLU
Test Loss	~0.073	~0.110	✅ ReLU
Convergence Speed	~2 epochs to 95%	~5 epochs to 95%	✅ ReLU
Training Time	Faster	Slower	✅ ReLU
Note: Exact values may vary slightly between runs due to random initialization.
🔑 Key Findings
1. ReLU Converges Faster
ReLU reached 95% validation accuracy approximately 2-3x faster than Sigmoid
because its gradient is either 0 or 1 — never a tiny fraction that would slow
down learning.

2. Sigmoid Suffers from Vanishing Gradients
Sigmoid's maximum gradient is only 0.25. When multiplied across multiple layers
during backpropagation, this creates the vanishing gradient problem — earlier
layers barely learn.

3. ReLU Achieves Higher Final Accuracy
ReLU's non-saturating nature allows it to optimize more effectively, reaching
approximately 1% higher accuracy than Sigmoid.

4. Both Models Achieve Good Performance
Even with these differences, both architectures achieve >96% accuracy,
demonstrating that simple feedforward networks can effectively solve digit
recognition.

📊 Visualizations
Model A: ReLU Performance
ReLU Performance

Model B: Sigmoid Performance
Sigmoid Performance

Side-by-Side Comparison
Comparison

Sample Predictions
Predictions

🛠️ Technologies Used
Technology	Purpose
Python 3.x	Core programming language
TensorFlow 2.x	Deep learning framework
Keras	High-level neural network API
NumPy	Numerical computations
Matplotlib	Data visualization
🔮 Future Improvements
 🔄 Experiment with Leaky ReLU, ELU, and Swish activation functions
 🖼️ Implement Convolutional Neural Networks (CNNs) for >99% accuracy
 🛡️ Add Dropout and L2 regularization to reduce overfitting
 📊 Test on Fashion-MNIST and CIFAR-10 datasets
 📉 Implement learning rate scheduling
 🌐 Build a web application for real-time digit recognition
 📱 Deploy model on mobile devices using TensorFlow Lite
 🔍 Add confusion matrix and per-class accuracy analysis
🤝 Contributing
Contributions are welcome! Here's how:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License — see the LICENSE
file for details.
MIT License

Copyright (c) 2026 [Muhammad Immad Ahmed Khan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
🙏 Acknowledgments
Yann LeCun — Creator of the MNIST dataset
TensorFlow/Keras Team — For the excellent deep learning framework
Deep Learning Community — For open-source resources and tutorials
📧 Contact
Your Name — immadkhan303@gmail.com

Project Link: https://github.com/immad-khan/Neural-Network-for-Digit-Classification

<p align="center"> ⭐ If you found this project helpful, please give it a star! ⭐ </p> 
