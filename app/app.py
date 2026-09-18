import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import CLASS_NAMES
from src.gradcam import GradCAM, overlay_gradcam
from src.inference import load_trained_model, predict_image


DISPLAY_NAMES = {
    "angular_leaf_spot": "Angular Leaf Spot",
    "bean_rust": "Bean Rust",
    "healthy": "Healthy",
}


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource
def load_model():
    return load_trained_model()


model, device = load_model()


st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a bean leaf image to classify it as "
    "**Angular Leaf Spot**, **Bean Rust**, or **Healthy**."
)


with st.sidebar:
    st.header("Model")

    st.markdown(
        """
        **Architecture:** MobileNetV3 Small  
        **Approach:** Transfer learning + fine-tuning  
        **Test accuracy:** 90.62%  
        **Test Macro F1:** 90.77%
        """
    )

    st.header("Confidence")

    st.write(
        "Predictions below 70% confidence are flagged "
        "as low-confidence results."
    )

    st.caption(
        "This project is an ML demonstration and should not be "
        "used as a substitute for professional plant-disease diagnosis."
    )


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is None:
    st.info("Upload an image to begin.")

else:
    image = Image.open(uploaded_file).convert("RGB")

    result = predict_image(
        image,
        model,
        device,
    )

    prediction = result["prediction"]
    confidence = result["confidence"]

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

    with col2:
        st.metric(
            "Predicted class",
            DISPLAY_NAMES[prediction],
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%",
        )

        if result["accepted"]:
            st.success("Prediction accepted.")
        else:
            st.warning(
                "Low-confidence prediction — human review recommended."
            )

        st.subheader("Class probabilities")

        for item in result["top_3"]:
            class_name = DISPLAY_NAMES[item["class"]]
            probability = item["confidence"]

            st.write(
                f"**{class_name}:** "
                f"{probability * 100:.1f}%"
            )

            st.progress(probability)

    st.divider()

    st.subheader("Grad-CAM explanation")

    st.write(
        "The heatmap highlights image regions that had the "
        "strongest influence on the model's prediction."
    )

    target_layer = model.features[-1]

    gradcam = GradCAM(
        model,
        target_layer,
    )

    class_index = CLASS_NAMES.index(
        prediction
    )

    cam, _ = gradcam.generate(
        image,
        device,
        class_index=class_index,
    )

    overlay = overlay_gradcam(
        image,
        cam,
    )

    gradcam.remove_hooks()

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original",
            use_container_width=True,
        )

    with col2:
        st.image(
            overlay,
            caption="Grad-CAM",
            use_container_width=True,
        )

    st.caption(
        "Grad-CAM indicates which regions influenced the neural network. "
        "It does not prove that those regions caused the prediction."
    )
