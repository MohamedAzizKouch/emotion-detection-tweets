# 🎭 Emotion Detection from Tweets

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow?logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![GCP](https://img.shields.io/badge/Google_Cloud_Run-Deployed-4285F4?logo=googlecloud)
![Accuracy](https://img.shields.io/badge/Best_Accuracy-92.60%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Deep Learning Project — 2025/2026**  
> A comparative study of CNN, BiLSTM, CNN-BiLSTM-Attention and DistilBERT for multi-class emotion classification of tweets into 6 categories: *joy, sadness, anger, fear, love, surprise*.

---

## 🌐 Live Demo

**🚀 Try the API:** [https://emotion-api-13019715482.us-central1.run.app](https://emotion-api-13019715482.us-central1.run.app)

**📖 Swagger Docs:** [https://emotion-api-13019715482.us-central1.run.app/docs](https://emotion-api-13019715482.us-central1.run.app/docs)

---

## 📊 Results Summary

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|-------|:------------:|:--------:|:-----------:|
| CNN Baseline | 90.05% | 0.86 | 0.90 |
| BiLSTM | 87.05% | 0.84 | 0.87 |
| CNN-BiLSTM-Attention | 87.95% | 0.85 | 0.89 |
| **DistilBERT** ⭐ | **92.60%** | **0.89** | **0.93** |

> DistilBERT outperforms all baselines by **+2.55%** over CNN and **+5.55%** over BiLSTM.

---

## 📁 Repository Structure

```
emotion-detection-tweets/
│
├── app.py                    # FastAPI application (REST API + Web UI)
├── predict.py                # Prediction logic (load model, tokenize, infer)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container configuration
│
├── notebooks/
│   ├── SentimentAnalysis.ipynb   # CNN, BiLSTM, CNN-BiLSTM-Attention
│   └── BERT.ipynb                # DistilBERT fine-tuning
│
├── model/                    # Model config & tokenizer files
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│   # ⚠️ model.safetensors hosted on Hugging Face (see below)
│
├── data/
│   └── README.md             # Dataset info & Kaggle link
│
└── report/
    └── DL_Rapport.pdf        # Full project report
```

---

## 🧠 Model Architecture

### Innovation: Fine-tuned DistilBERT

```
Raw Tweet Text
      ↓
WordPiece Tokenizer → [CLS] tokens [SEP]
      ↓
DistilBERT Embeddings (pre-trained on 11GB text)
      ↓
6 Transformer Layers (self-attention + FFN)
      ↓
[CLS] Token Representation (768-dim)
      ↓
Classification Head: Dense(768 → 6) + Softmax
      ↓
6 Emotion Probabilities
```

### Baselines
- **CNN**: Multi-filter 1D CNN (filters: 2, 3, 4) + Global Max Pooling — 2.7M params
- **BiLSTM**: 2 stacked Bidirectional LSTM layers (128+64 units) — 3.0M params
- **CNN-BiLSTM-Attention**: Hybrid with custom trainable attention mechanism — 2.9M params

---

## 📦 Dataset

**Emotion Dataset for NLP** — [Kaggle Link](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)

| Split | Samples |
|-------|--------:|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |
| **Total** | **20,000** |

6 emotion classes: `joy (33.5%)` · `sadness (29.2%)` · `anger (13.5%)` · `fear (12.1%)` · `love (8.2%)` · `surprise (3.6%)`

> Class imbalance handled with **class-weighted loss** in all models.

---

## ⚙️ Installation & Local Run

### 1. Clone the repository

```bash
git clone https://github.com/MohamedAzizKouch/emotion-detection-tweets.git
cd emotion-detection-tweets
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model weights

The `model.safetensors` file (~250MB) is hosted on Hugging Face:

```bash
# Option A — using huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='AzizKouch/emotion-detection-distilbert',
    filename='model.safetensors',
    local_dir='model/'
)
"
```

Or download manually from: [Hugging Face Model Page](hhttps://huggingface.co/AzizKouch/emotion-detection-distilbert)

### 4. Run the API locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🐳 Docker

```bash
# Build the image
docker build -t emotion-api .

# Run the container
docker run -p 8000:8000 emotion-api
```

---

## 🔌 API Usage

### Single prediction

```bash
curl -X POST "https://emotion-api-13019715482.us-central1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so happy today, life is wonderful!"}'
```

**Response:**
```json
{
  "emotion": "joy",
  "confidence": 0.9823,
  "probabilities": {
    "joy": 0.9823,
    "sadness": 0.0054,
    "anger": 0.0031,
    "fear": 0.0028,
    "love": 0.0042,
    "surprise": 0.0022
  }
}
```

### Batch prediction

```bash
curl -X POST "https://emotion-api-13019715482.us-central1.run.app/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["I feel lonely", "This is amazing!", "I am so angry right now"]}'
```

### Health check

```bash
curl https://emotion-api-13019715482.us-central1.run.app/health
```

---

## ☁️ Cloud Deployment

Deployed on **Google Cloud Run** (serverless, auto-scaling):

| Component | Details |
|-----------|---------|
| Framework | FastAPI + Uvicorn |
| Container | Docker (Python 3.10-slim) |
| Platform | Google Cloud Run (us-central1) |
| Memory | 4 GiB |
| CPU | 2 vCPU |
| Access | Public (unauthenticated) |

### Deploy your own instance

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/emotion-api

# Deploy to Cloud Run
gcloud run deploy emotion-api \
  --image gcr.io/YOUR_PROJECT_ID/emotion-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --allow-unauthenticated
```

---

## 🛠️ Training Configuration

| Hyperparameter | CNN | BiLSTM | CNN+BiLSTM+Att. | DistilBERT |
|----------------|:---:|:------:|:---------------:|:----------:|
| Optimizer | Adam | Adam | Adam | AdamW |
| Learning Rate | 1e-3 | 1e-3 | 1e-3 | 2e-5 |
| Batch Size | 64 | 64 | 64 | 16 |
| Max Epochs | 20 | 20 | 20 | 4 |
| Early Stopping | pat.=3 | pat.=3 | pat.=3 | pat.=2 |
| Max Seq. Length | 50 | 50 | 50 | 64 |
| GPU | NVIDIA GeForce GTX 1650 Ti |

---

## 📚 References

- Sanh et al. (2019) — [DistilBERT](https://arxiv.org/abs/1910.01108)
- Devlin et al. (2019) — [BERT](https://arxiv.org/abs/1810.04805)
- Kim (2014) — [CNN for Sentence Classification](https://arxiv.org/abs/1408.5882)
- Saravia et al. (2018) — [CARER Dataset](https://aclanthology.org/D18-1404/)

---

## 👨‍🎓 Author

**[Mohamed Aziz Kouch]**  
Deep Learning Project — Faculty of Sciences, 2025/2026  
Supervisor: [Mr.Ahmed Ben Taleb]

---

## 📄 License

This project is licensed under the MIT License.
