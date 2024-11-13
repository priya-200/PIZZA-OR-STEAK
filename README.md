# 🍕 Pizza vs Steak: Image Classification with TensorFlow 🍖

This project involves building an image classification model to classify images of either **Pizza** or **Steak** using **TensorFlow**. The project demonstrates both a **custom CNN model** and the use of **TensorFlow Hub pre-trained models** for efficient transfer learning.

---

## 🚀 Project Overview  
- **Objective**: Classify images into two categories: Pizza and Steak.
- **Approach**:
  - **Custom Model**: Built a Convolutional Neural Network (CNN) model using TensorFlow.
  - **Transfer Learning with TensorFlow Hub**: Utilized pre-trained models from TensorFlow Hub for faster and more accurate results with less training data.
  
- **Dataset**: The dataset consists of images of pizzas and steaks, collected from a publicly available image dataset or manually curated.

---

## 🛠️ Tools and Technologies  
- **Python** 🐍  
- **TensorFlow** and **Keras** for deep learning model development  
- **TensorFlow Hub** for pre-trained models  
- **Matplotlib** and **Seaborn** for data visualization  
- **NumPy** for numerical computations  
- **Pandas** for data management  
- **OpenCV** for image processing  

---

## 📊 Dataset Details  
- **Source**: The dataset is a collection of images of pizzas and steaks.
- **Classes**:  
  - **Pizza**  
  - **Steak**

- **Image Dimensions**: Images are resized to a uniform size (e.g., 224x224 pixels) for feeding into the model.
  
---

## 🧠 Model Overview  
1. **Custom CNN Model**:  
   - The custom model uses a series of convolutional layers, activation functions (ReLU), and pooling layers to extract features from images.
   - A fully connected layer followed by a softmax layer for classification into either pizza or steak.

2. **Transfer Learning with TensorFlow Hub**:  
   - The pre-trained models are loaded from TensorFlow Hub and fine-tuned on the pizza vs steak dataset.
   - This method allows for leveraging the knowledge from large image datasets, speeding up the training process and improving accuracy with less data.

3. **Model Evaluation**:  
   - The models are evaluated using **accuracy**, **precision**, **recall**, and **confusion matrix**.
   - **Loss** and **accuracy curves** are plotted for model performance tracking.

---

## 🏆 Results  
- **Accuracy**: The custom CNN model and TensorFlow Hub-based model show impressive classification accuracy on the test set.
- **Evaluation Metrics**:  
  - **Accuracy**: Measures the percentage of correct classifications.  
  - **Precision & Recall**: Evaluate the model's performance in terms of false positives and false negatives.  
  - **Confusion Matrix**: Provides insights into misclassifications.

- **Comparison**: Results are compared between the custom model and TensorFlow Hub model to demonstrate the impact of transfer learning.

---
