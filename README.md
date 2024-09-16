# Potato Disease Classifier

## Project Overview

This project is a **Potato Disease Classifier** end-to-end application built using **FastAPI** as the backend framework and **TensorFlow** to serve a deep learning model that classifies potato leaf diseases and **ReactJs** for frontend. The application predicts one of three classes:
1. Early Blight
2. Late Blight
3. Healthy

The application allows users to upload an image of a potato leaf, and it returns the predicted class along with a confidence score. You can check it out here: https://potato-disease-frontend-3glo320rl.vercel.app/

---

## Features

- **API Backend**: A FastAPI-based backend that accepts image uploads and returns disease classification.
- **TensorFlow Model**: A trained convolutional neural network (CNN) that predicts whether a potato leaf is healthy or diseased (Early Blight or Late Blight).
- **CORS**: CORS middleware is configured to enable cross-origin communication, useful for frontend integration.
- **Preprocessing & Model Training**: A detailed model training process is captured in the Jupyter notebook `potato_disease_classifier.ipynb`.


### Explanation of `potato_disease_classifier.ipynb` (Model Training)

In the `potato_disease_classifier.ipynb` file, the following steps were executed to build and train the model:

1. **Data Loading & Preprocessing**
   - **Dataset:** A dataset containing images of potato leaves categorized into three classes: Early Blight, Late Blight, and Healthy.
   - **Data Augmentation:** Image data was augmented using techniques such as random flipping and rotation to increase the variety of training samples and improve model generalization.
   - **Normalization:** Pixel values were normalized to a range of [0, 1] by dividing by 255.

2. **Model Architecture**
   - The model was built using Convolutional Neural Networks (CNN) in TensorFlow/Keras.
   - The architecture consists of multiple `Conv2D` layers followed by `MaxPooling2D` layers, culminating in a `Flatten` layer and fully connected `Dense` layers.
   - The model ends with a `softmax` layer for multi-class classification.

3. **Model Training**
   - **Loss Function:** `SparseCategoricalCrossentropy`, suitable for multi-class classification with integer labels.
   - **Optimizer:** `Adam`, a widely-used optimizer for training neural networks.
   - **Metrics:** Accuracy was used to evaluate model performance during training.
![potato_model_chart](https://github.com/user-attachments/assets/638d02c1-7c35-4f6d-b10f-d65ac80c603e)


---

## Requirements

The project requires the following Python libraries, listed in `requirements.txt`:

```plaintext
tensorflow
fastapi
uvicorn
python-multipart
pillow
matplotlib
numpy
