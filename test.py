import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")

text = "This movie was terrible. I hated it."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    probs = torch.nn.functional.softmax(logits, dim=1)

print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Predicted class: {predicted_class}")
