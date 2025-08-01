import streamlit as st
import onnxruntime
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import io
import base64
import time

# Placeholder for class names - REPLACE WITH YOUR ACTUAL 12 CLASS NAMES
CLASS_NAMES = ['ripe apple', 'ripe banana', 'ripe dragon', 'ripe grapes', 'ripe lemon', 'ripe mango',
          'ripe orange', 'ripe papaya', 'ripe pineapple', 'ripe pomegranate', 'ripe strawberry',
           'unripe apple', 'unripe banana', 'unripe dragon', 'unripe grapes', 'unripe lemon',
           'unripe mango', 'unripe orange', 'unripe papaya', 'unripe pineapple', 'unripe pomegranate',
           'unripe strawberry'] 

def load_model(model_path):
    """Loads the ONNX model."""
    try:
        session = onnxruntime.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

def preprocess_image(image):
    """
    Preprocesses the uploaded image for model inference.
    
    This function assumes the model expects a 224x224 RGB image,
    normalized with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    Adjust these parameters according to your model's training.
    """
    image = image.convert("RGB")
    image = image.resize((224, 224)) # Resize to expected input size
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0 # Normalize to [0, 1]

    # Standardize with mean and std (ImageNet values, common for pre-trained models)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_array = np.transpose(img_array, (2, 0, 1)) # Change from HWC to CHW
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array.astype(np.float32) # Ensure final output is float32

    return img_array

def main():
    st.set_page_config(page_title="Fruit Classifier", layout="wide")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the app mode",
        ["Image Classification", "Webcam Classification", "About"])

    st.sidebar.markdown("--- App Settings ---")
    top_n = st.sidebar.slider("Show Top N Predictions", 1, len(CLASS_NAMES), 5)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Only show predictions above this confidence level.")

    st.title("🍎🍇🍊 Fruit Classifier 🍋🍓🥭")

    st.markdown("""
    Welcome to the Fruit Classifier! 
    Upload an image or use your webcam to classify different types of fruits.
    This model can distinguish between 22 different fruit categories.
    """)

    model_path = "fruit_classifier_model.onnx"
    session = load_model(model_path)

    if not session:
        st.warning("Model could not be loaded. Please ensure 'fruit_classifier_model.onnx' is in the correct directory.")
        return

    if app_mode == "Image Classification":
        st.header("Upload an Image")
        image_classification_mode(session, top_n, confidence_threshold)
    elif app_mode == "Webcam Classification":
        st.header("Live Webcam Classification")
        webcam_classification_mode(session, top_n, confidence_threshold)
    elif app_mode == "About":
        show_about_section()

def image_classification_mode(session, top_n, confidence_threshold):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        
        with st.spinner("Classifying..."):
            classified_image = classify_image(image, session, top_n, confidence_threshold)
            if classified_image:
                st.download_button(
                    label="Download Classified Image",
                    data=classified_image,
                    file_name="classified_fruit.png",
                    mime="image/png"
                )

def webcam_classification_mode(session, top_n, confidence_threshold):
    st.info("Grant access to your webcam. Press 'Stop Webcam' to end the feed.")
    
    st.sidebar.markdown("--- Webcam Settings ---")
    resolution_options = {"640x480": (640, 480), "1280x720": (1280, 720)}
    selected_resolution_str = st.sidebar.selectbox("Resolution", list(resolution_options.keys()))
    selected_resolution = resolution_options[selected_resolution_str]

    stop_button = st.button("Stop Webcam")

    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Make sure no other application is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, selected_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, selected_resolution[1])

    st_frame = st.empty()
    fps_text = st.empty()

    prev_frame_time = 0

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame. Exiting webcam stream.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        st_frame.image(pil_image, channels="RGB", use_column_width=True)

        with st.spinner("Classifying frame..."):
            classified_image_bytes = classify_image(pil_image, session, top_n, confidence_threshold)
            if classified_image_bytes: # Check if classification was successful
                st.download_button(
                    label="Download Last Classified Frame",
                    data=classified_image_bytes,
                    file_name="classified_frame.png",
                    mime="image/png",
                    key="download_webcam_frame"
                )

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text.text(f"FPS: {int(fps)}")

        time.sleep(0.05)

        # Use Streamlit's session state to manage the stop button more reliably
        # This prevents the button from being re-rendered and losing its state on each loop iteration.
        if st.session_state.get("stop_webcam_clicked", False):
            break # Exit the loop if the stop button was truly clicked
        
    cap.release()
    st.session_state["stop_webcam_clicked"] = False # Reset the state
    st.success("Webcam stream stopped.")

def classify_image(image, session, top_n, confidence_threshold):
    preprocessed_image = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    try:
        outputs = session.run([output_name], {input_name: preprocessed_image})
        probabilities = outputs[0][0]

        exp_probs = np.exp(probabilities - np.max(probabilities))
        softmax_probs = exp_probs / np.sum(exp_probs)
        
        # Get top N predictions that meet the confidence threshold
        qualified_indices = np.where(softmax_probs >= confidence_threshold)[0]
        if len(qualified_indices) == 0:
            st.warning("No prediction met the confidence threshold. Try a lower threshold or a different image.")
            return None

        # Sort qualified predictions by confidence and get top N
        sorted_qualified_indices = qualified_indices[np.argsort(softmax_probs[qualified_indices])[::-1]]
        top_n_indices = sorted_qualified_indices[:top_n]

        top_n_classes = [CLASS_NAMES[i] for i in top_n_indices]
        top_n_confidences = [softmax_probs[i] * 100 for i in top_n_indices]

        # Display top N predictions in a table
        results_df = pd.DataFrame({
            'Fruit': top_n_classes,
            'Confidence': [f'{c:.2f}%' for c in top_n_confidences]
        })
        st.subheader("Top Predictions:")
        st.table(results_df)

        # Add a confidence bar chart for all classes
        chart_data = pd.Series(softmax_probs, index=CLASS_NAMES)
        st.bar_chart(chart_data)

        # Return image as bytes for download
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

def show_about_section():
    st.header("About the Fruit Classifier")
    st.markdown("""
    This application uses a deep learning model to classify images of fruits into 22 different categories.
    
    **Model Details:**
    *   **Architecture:** ResNet-like (or similar, depending on your `fruit_classifier_model.onnx`)
    *   **Training Data:** [You can add details about your training dataset here, e.g., 'a custom dataset of fruit images']
    *   **Number of Classes:** 22 (as defined in `CLASS_NAMES`)
    *   **Purpose:** To demonstrate real-time image classification using Streamlit and ONNX.
    
    **How to Use:**
    1.  **Image Classification:** Upload an image file (JPG, JPEG, PNG) to get a prediction.
    2.  **Webcam Classification:** Use your device's webcam for live fruit classification. Grant camera permissions when prompted.
    
    **Developed by:** Your Name/Organization (Optional)
    """)

    st.markdown("--- Version 1.0 --- ")

if __name__ == "__main__":
    main() 