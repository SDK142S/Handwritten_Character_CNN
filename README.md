# Handwritten Character Recognition using Convolutional Neural Networks (CNN)

This project is a **Handwritten Character Recognition** system built using Convolutional Neural Networks (CNN). It recognizes handwritten alphabets (A-Z) and classifies them. The project includes two main components:
- **`character_recognition.ipynb`**: A Jupyter notebook that contains the model building, training, and evaluation for the CNN.
- **`project.py`**: A Streamlit application that allows users to interact with the model and recognize handwritten characters through a web interface.

---

## Project Overview

The Handwritten Character Recognition system utilizes Convolutional Neural Networks (CNN) to recognize **handwritten English alphabets (A-Z)**. It is trained on the **A-Z Handwritten Alphabets dataset** and is capable of classifying new handwritten input.

The project also features a **Streamlit** web application that allows users to:
- Draw a character on a canvas.
- Predict the character using the trained CNN model.
- Display the result and confidence score.

---

## Files Description

### `character_recognition.ipynb`

This Jupyter notebook handles:
- **Data Preprocessing**: Loads and processes the A-Z Handwritten Alphabets dataset.
- **Model Building**: Builds and trains a CNN model using TensorFlow/Keras.
- **Model Evaluation**: Evaluates the model using accuracy and confusion matrix.

Key Features:
- Load the dataset.
- Normalize and reshape the data.
- Build, compile, and train the CNN.
- Save the trained model weights for use in the Streamlit app.

### `project.py`

A **Streamlit** application that:
- Allows users to draw a character.
- Predicts the character using the trained CNN model.
- Displays the predicted character and the model's confidence score.

---

## Dataset

The dataset used in this project is the **A-Z Handwritten Alphabets dataset** available on Kaggle. This dataset contains over **370,000 images** of handwritten English alphabets (A-Z), where each character image is 28x28 pixels.

- **Dataset Name**: A-Z Handwritten Alphabets (CSV Format)
- **Source**: [A-Z Handwritten Alphabets on Kaggle](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

This dataset is used to train the CNN model to recognize handwritten characters.

---

## Sample UI

Here’s a sample image of the **Streamlit UI** where users can interact with the model:

![Sample UI](https://github.com/your-username/Handwritten_Character_CNN/blob/main/sample.png)
