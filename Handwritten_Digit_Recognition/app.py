
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

IMG_SIZE = 128

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "AI-based handwritten digit recognition using "
    "a Convolutional Neural Network."
)

st.info("Upload a handwritten digit from 0 to 9.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")


# Image preprocessing
def preprocess_image(image):

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    arr = np.array(image)

    # Detect background using border pixels
    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ])

    # Invert if background is brighter
    if border.mean() > arr.mean():
        image = ImageOps.invert(image)

    # Remove low-intensity noise
    image = image.point(
        lambda p: 255 if p > 40 else 0
    )

    # Find digit boundaries
    bbox = image.getbbox()

    if bbox is None:
        return None

    image = image.crop(bbox)

    # Resize digit while maintaining aspect ratio
    image.thumbnail(
        (92, 92),
        Image.Resampling.LANCZOS
    )

    # Create 128x128 black canvas
    canvas = Image.new(
        "L",
        (IMG_SIZE, IMG_SIZE),
        0
    )

    # Center digit
    x = (IMG_SIZE - image.width) // 2
    y = (IMG_SIZE - image.height) // 2

    canvas.paste(image, (x, y))

    # Normalize image
    image_array = (
        np.array(canvas).astype("float32") / 255.0
    )

    # Center by center of mass
    total = image_array.sum()

    if total > 0:

        rows, cols = np.indices(image_array.shape)

        center_y = (
            rows * image_array
        ).sum() / total

        center_x = (
            cols * image_array
        ).sum() / total

        shift_y = round(63.5 - center_y)
        shift_x = round(63.5 - center_x)

        image_array = np.roll(
            image_array,
            (shift_y, shift_x),
            axis=(0, 1)
        )

        # Remove wrapped pixels
        if shift_y > 0:
            image_array[:shift_y, :] = 0
        elif shift_y < 0:
            image_array[shift_y:, :] = 0

        if shift_x > 0:
            image_array[:, :shift_x] = 0
        elif shift_x < 0:
            image_array[:, shift_x:] = 0

    # Model input shape: 1x128x128x1
    image_array = image_array.reshape(
        1, IMG_SIZE, IMG_SIZE, 1
    )

    return image_array, canvas


# Upload image
uploaded_file = st.file_uploader(
    "Choose a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(original_image, width=250)

    processed = preprocess_image(original_image)

    if processed is None:

        st.error(
            "No digit detected. Please upload another image."
        )

    else:

        image_array, processed_image = processed

        st.subheader("128x128 Processed Image")

        st.image(
            processed_image,
            width=256
        )

        try:

            with st.spinner("AI is recognising the digit..."):

                model = load_model()

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                predicted_digit = int(
                    np.argmax(predictions)
                )

                confidence = float(
                    np.max(predictions)
                ) * 100

            st.success("Prediction completed!")

            st.metric(
                "Recognised Digit",
                predicted_digit
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.subheader("Digit Prediction Probabilities")

            probabilities = {
                str(i): float(predictions[0][i])
                for i in range(10)
            }

            st.bar_chart(probabilities)

        except Exception as e:

            st.error(f"Prediction error: {e}")

else:

    st.warning("Upload an image to begin recognition.")

st.divider()

st.caption(
    "Class 12 Artificial Intelligence Project | CNN"
)
