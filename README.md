# 🧠 Breast Cancer Classification – From Scratch to Deep Learning

This repository demonstrates **three different approaches** to solving a **binary classification problem** using the **Breast Cancer Wisconsin Diagnostic Dataset**.

The project progresses from **classical machine learning implemented from scratch**, to **deep learning using frameworks**, and finally to a **fully custom deep neural network with Adam optimization built using NumPy only**.

---

## 🔹 Model 1: Logistic Regression from Scratch (NumPy)

### 📌 Description
A **binary logistic regression classifier** implemented entirely using **NumPy**.  
This model uses **batch gradient descent** and a **sigmoid activation function** to predict cancer malignancy.

### ⚙️ Key Details
- Optimization: Batch Gradient Descent  
- Learning Rate: `0.01`  
- Iterations: `1000`  
- Feature Scaling: `StandardScaler`  
- Threshold: `0.5`

### 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### ✅ Accuracy
- **Test Accuracy: 0.9912280701754386**  


---

## 🔹 Model 2: Neural Network using TensorFlow / Keras

### 📌 Description
A **feedforward Artificial Neural Network (ANN)** built using **TensorFlow/Keras**, demonstrating how deep learning frameworks simplify model development while achieving high performance.

### 🧠 Architecture
_______________________________
| Layer  | Units | Activation |
|--------|-------|------------|
|  Input |  30   |    ReLU    |
| Hidden |   8   |    ReLU    |
| Output |   1   |   Sigmoid  |
_______________________________
### ⚙️ Training Setup
- Optimizer: Adam  
- Loss Function: Binary Cross-Entropy  
- Epochs: Up to 100  
- Early Stopping: Stops when accuracy exceeds **98%**  
- Feature Scaling: Min–Max Normalization  

### 📊 Evaluation Metrics
- Accuracy & Loss  
- Classification Report  
- Confusion Matrix  
- Accuracy vs Epoch plots  
- Confusion Matrix Heatmap  

### ✅ Accuracy
- **Test Accuracy: 0.9649122953414917**

---

## 🔹 Model 3: Deep Neural Network from Scratch (NumPy + Adam)

### 📌 Description
A **fully custom deep neural network** implemented using **only NumPy**, including:

- Forward propagation  
- Backpropagation  
- Binary cross-entropy loss  
- Adam optimizer with bias correction  
- Custom classification report  
- Custom confusion matrix  

### 🧠 Architecture
_______________________________________

| Layer          | Units | Activation |
|----------------|-------|------------|
| Input          |  30   |      —     |
| Hidden Layer 1 |  30   |    ReLU    |
| Hidden Layer 2 |   8   |    ReLU    |
| Output         |   1   |   Sigmoid  |
________________________________________
### ⚙️ Optimization
- Optimizer: Adam (from scratch)  
- Learning Rate: `0.001`  
- Iterations: `5000`  
- β₁ = `0.9`, β₂ = `0.999`  
- ε = `1e-8`  

### 📊 Evaluation Metrics
- Custom Precision, Recall, F1-score  
- Custom Confusion Matrix  
- Accuracy  

### ✅ Accuracy
- **Test Accuracy: 0.9736842105263158**

---

## 📈 Performance Comparison
___________________________________________________________________

|        Model        |   Implementation     |      Accuracy      |
|---------------------|--------------------- |--------------------|
| Logistic Regression | NumPy (from scratch) | 0.9912280701754386 |
| Neural Network      | TensorFlow / Keras   | 0.9649122953414917 |
| Deep Neural Network | NumPy + Adam         | 0.9736842105263158 |
___________________________________________________________________
---
 

