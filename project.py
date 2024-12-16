import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(
    page_title="Handwritten Character Recognition",
    layout="wide",  # Enables wide layout
    initial_sidebar_state="collapsed",
)

# Load your model
model = load_model('best_model.keras')

# Dictionary mapping predictions to letters
dict_word = {i: chr(65 + i) for i in range(26)}  # A-Z mapping (0->A, 1->B, ..., 25->Z)

# Function to preprocess the image
def preprocess_image(image):
    # Convert the PIL Image to a NumPy array
    image = np.array(image)

    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply median blur
    gray = cv.medianBlur(gray, 5)

    # Apply thresholding
    _, gray = cv.threshold(gray, 75, 180, cv.THRESH_BINARY)

    # Apply morphological gradient
    element = cv.getStructuringElement(cv.MORPH_RECT, (90, 90))
    gray = cv.morphologyEx(gray, cv.MORPH_GRADIENT, element)

    # Downsample the image
    gray = gray / 255.0

    # Resize to 28x28
    gray = cv.resize(gray, (28, 28))

    # Reshape for the model
    gray = np.reshape(gray, (1, 28, 28, 1))

    return gray

# Apply custom CSS for matte black background and text styling
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;
        color: #ffffff;
    }
    .stApp {
        background-color: #0E1117;
        padding: 20px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    .prediction-box {
    background-color: rgba(0, 128, 0, 0.4); /* Matte green with increased transparency */
    color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("Handwritten Character Recognition")

# Divide the UI into two columns (left and right)
left_column, right_column = st.columns(2)

# Left column: Drag-and-drop box for uploading images
with left_column:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Resize the image to a fixed size for display
        fixed_size_image = image.resize((12 * 28, 12 * 28))  # 12x12 inches at 28px per inch
        st.image(fixed_size_image, caption="Uploaded Image", output_format="auto")
    else:
        st.text("Upload an image to display here")

# Right column: Prediction box
with right_column:
    st.header("Prediction")
    if uploaded_file is not None:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict using the model
        prediction = np.argmax(model.predict(processed_image))
        predicted_letter = dict_word[prediction]

        # Display the prediction
        st.markdown(
            f"""
            <div class='prediction-box'>
                Prediction: "{predicted_letter}"
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class='prediction-box'>
            <h1></h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
