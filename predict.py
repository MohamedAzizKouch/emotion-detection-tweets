import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Constantes ──────────────────────────────────────────────────
EMOTION_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

EMOTION_EMOJI = {
    "sadness" : "😢",
    "joy"     : "😊",
    "love"    : "❤️",
    "anger"   : "😠",
    "fear"    : "😨",
    "surprise": "😲"
}

MAX_LEN    = 64
MODEL_PATH = "./model"

# ── Chargement du modèle (une seule fois au démarrage) ──────────
print("Loading DistilBERT model...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model     = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

# ── Fonction de prédiction ───────────────────────────────────────
def predict_emotion(text: str) -> dict:
    # Tokenisation
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inférence
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)
        probs   = probs.cpu().numpy()[0]

    pred_label = int(np.argmax(probs))
    emotion    = EMOTION_MAP[pred_label]

    return {
        "text"       : text,
        "emotion"    : emotion,
        "emoji"      : EMOTION_EMOJI[emotion],
        "confidence" : round(float(probs[pred_label]) * 100, 2),
        "all_probabilities": {
            EMOTION_MAP[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
    }

def predict_batch(texts: list) -> list:
    return [predict_emotion(t) for t in texts]