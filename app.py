
import streamlit as st
from pathlib import Path

from src.model_loader import load_model
from src.predictor import predict_hallucination


# Page configuration
st.set_page_config(
    page_title="RAGGuard | Hallucination Detector",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = Path("Models/modernbert_final")


# Load model only once
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


# Header
st.title("🛡️ RAGGuard")
st.caption(
    "Transformer-Based Framework for Detecting "
    "Hallucinations in Retrieval-Augmented Generation Systems"
)

st.divider()

st.subheader("Analyze an AI-generated response")

st.write(
    "Evaluate whether an AI-generated response is supported "
    "by the provided source context using ModernBERT."
)

# Input areas
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Source Context")
    source_text = st.text_area(
        "Source Context",
        height=300,
        placeholder="Paste the retrieved source passage here...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("🤖 Generated Response")
    response_text = st.text_area(
        "Generated Response",
        height=300,
        placeholder="Paste the AI-generated response here...",
        label_visibility="collapsed"
    )

st.write("")

# Analyze
if st.button("🔍 Analyze Response", use_container_width=True):

    if not source_text.strip() or not response_text.strip():
        st.warning("Please enter both source context and generated response.")

    else:
        try:
            with st.spinner("Loading ModernBERT and analyzing response..."):

                model, tokenizer = get_model()

                result = predict_hallucination(
                    source_text=source_text,
                    response_text=response_text,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=4096
                )

            st.divider()
            st.subheader("Analysis Results")

            if result["prediction"] == "Hallucinated":
                st.error("⚠️ Hallucinated Response Detected")
            else:
                st.success("✅ Response Supported by Source")

            confidence = result["confidence"] * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(result["confidence"])

            st.subheader("Class Probabilities")

            col3, col4 = st.columns(2)

            with col3:
                st.metric(
                    "Supported",
                    f'{result["probabilities"]["supported"] * 100:.2f}%'
                )

            with col4:
                st.metric(
                    "Hallucinated",
                    f'{result["probabilities"]["hallucinated"] * 100:.2f}%'
                )

            st.caption(
                "Predictions indicate consistency with the supplied source. "
                "They are not independent verification of factual truth."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


st.divider()

st.caption(
    "RAGGuard | ModernBERT | RAG Hallucination Detection"
)