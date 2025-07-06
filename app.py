import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

# Define the path to the saved model and tokenizer directory
MODEL_DIR = "sentiment_model"

# Function to load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory '{MODEL_DIR}' not found.")
        return None, None
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Predict sentiment
def predict_sentiment(text, model, tokenizer):
    if model is None or tokenizer is None:
        return "Model or tokenizer not loaded."

    inputs = tokenizer(
        text, padding='max_length', truncation=True,
        max_length=256, return_tensors='pt'
    )

    device = torch.device('cpu')
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    return 'positive' if predicted_class == 1 else 'negative'

# Streamlit UI
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below:")

review_text = st.text_area("Enter review here:", height=200)

if st.button("Analyze Sentiment"):
    if review_text.strip():
        model, tokenizer = load_model_and_tokenizer()
        if model and tokenizer:
            sentiment = predict_sentiment(review_text, model, tokenizer)
            st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review to analyze.")