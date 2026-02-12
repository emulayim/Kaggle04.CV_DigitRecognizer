import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from PIL import Image, ImageOps
import tensorflow as tf

# --- Page Config ---
st.set_page_config(
    page_title="Digit Recognizer (MNIST)",
    page_icon="🔢",
    layout="wide"
)

# --- Helper Functions ---
def resolve_model_path(filename):
    possible_paths = [
        os.path.join("models", filename),
        os.path.join("src", filename),
        filename,
        os.path.join("..", "models", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def load_model():
    # Priority: Keras model (CNN) -> then sklearn if needed, but we prefer CNN here
    model_path = resolve_model_path("best_model.keras")
    if model_path:
        return tf.keras.models.load_model(model_path)
    return None

def preprocess_image(image):
    """
    Converts uploaded image to 28x28 grayscale, normalized array.
    """
    # Convert to grayscale
    img = ImageOps.grayscale(image)
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Invert colors if needed (MNIST is white digit on black background)
    # Usually users upload black digit on white background -> Invert
    # Simple heuristic: check mean pixel intensity.
    img_array = np.array(img)
    if img_array.mean() > 127: # Likely white background
        img = ImageOps.invert(img)
    
    # Normalize (0-255 -> 0-1)
    img_array = np.array(img).astype('float32') / 255.0
    
    # Reshape for CNN (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, img

# --- Main App ---
def main():
    st.title("🔢 Digit Recognizer (MNIST)")
    st.markdown("""
    **[TR]** El yazısı rakamları (0-9) tanıyan Yapay Zeka modeli.
    **[EN]** AI model to recognize handwritten digits (0-9).
    """)

    model = load_model()

    if model is None:
        st.error("🚨 Model file (`best_model.keras`) not found! Please run the notebook to train the model first.")
        return

    tab1, tab2 = st.tabs(["📁 Batch Prediction (CSV)", "✍️ Single Image Upload"])

    # --- TAB 1: CSV Upload ---
    with tab1:
        st.subheader("Upload Test Data (CSV)")
        st.info("Format: 784 columns (pixel0, pixel1, ... pixel783)")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("First 5 rows:")
                st.dataframe(df.head())

                if st.button("Predict Batch"):
                    # Preprocess for CNN: Reshape (N, 784) -> (N, 28, 28, 1) & Normalize
                    # Assuming data is 0-255
                    X = df.values.astype('float32') / 255.0
                    X = X.reshape(-1, 28, 28, 1)
                    
                    # Predict
                    preds_proba = model.predict(X)
                    preds_classes = np.argmax(preds_proba, axis=1)
                    
                    # Results
                    results = pd.DataFrame()
                    results["ImageId"] = range(1, len(df) + 1)
                    results["Label"] = preds_classes
                    
                    st.success("✅ Prediction Complete!")
                    
                    # Visualization
                    fig = px.histogram(results, x="Label", title="Predicted Label Distribution", nbins=10)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Submission CSV",
                        data=csv,
                        file_name="submission.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error: {e}")

    # --- TAB 2: Image Upload ---
    with tab2:
        st.subheader("Upload a Single Digit Image")
        img_file = st.file_uploader("Upload Image (PNG, JPG)", type=["png", "jpg", "jpeg"])
        
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", width=150)
            
            if st.button("Identify Digit"):
                # Preprocess
                processed_img, display_img = preprocess_image(image)
                
                # Show processed view (what the model sees)
                st.image(display_img, caption="Processed (28x28 Grayscale)", width=150)
                
                # Predict
                proba = model.predict(processed_img)
                prediction = np.argmax(proba)
                confidence = np.max(proba)
                
                # Display Result
                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Predicted Digit", str(prediction))
                c1.metric("Confidence", f"{confidence:.2%}")
                
                # Bar Chart of Probabilities
                probs_df = pd.DataFrame({
                    "Digit": range(10),
                    "Probability": proba[0]
                })
                
                with c2:
                    fig = px.bar(probs_df, x="Digit", y="Probability", title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
