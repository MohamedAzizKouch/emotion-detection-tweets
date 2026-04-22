from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from predict import predict_emotion, predict_batch
from typing import List
import uvicorn

app = FastAPI(
    title       = "Emotion Detection API",
    description = "Detect emotions from tweets using fine-tuned DistilBERT",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Schemas ─────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 500:
            raise ValueError('Text too long (max 500 characters)')
        return v

class BatchInput(BaseModel):
    texts: List[str]

    @validator('texts')
    def batch_size_limit(cls, v):
        if len(v) == 0:
            raise ValueError('Texts list cannot be empty')
        if len(v) > 32:
            raise ValueError('Max 32 texts per batch')
        return v

# ── HTML Interface ───────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Emotion Detection API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f5f5f5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }

    .container {
      width: 100%;
      max-width: 680px;
    }

    /* Header */
    .header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .header h1 {
      font-size: 26px;
      font-weight: 600;
      color: #1a1a1a;
      margin-bottom: 6px;
    }
    .header p {
      font-size: 14px;
      color: #666;
    }
    .badge {
      display: inline-block;
      background: #e8f4e8;
      color: #2d7a2d;
      font-size: 12px;
      font-weight: 500;
      padding: 3px 10px;
      border-radius: 20px;
      margin-bottom: 10px;
    }

    /* Input Area */
    .input-card {
      background: #fff;
      border-radius: 12px;
      border: 1px solid #e5e5e5;
      padding: 1.25rem;
      margin-bottom: 1rem;
    }
    .input-label {
      font-size: 13px;
      color: #666;
      margin-bottom: 8px;
    }
    .input-row {
      display: flex;
      gap: 8px;
    }
    input[type="text"] {
      flex: 1;
      height: 42px;
      padding: 0 14px;
      font-size: 15px;
      border: 1px solid #ddd;
      border-radius: 8px;
      outline: none;
      color: #1a1a1a;
      background: #fafafa;
      transition: border-color 0.2s;
    }
    input[type="text"]:focus {
      border-color: #4a7fcb;
      background: #fff;
    }
    .analyze-btn {
      height: 42px;
      padding: 0 20px;
      background: #1a1a1a;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
      white-space: nowrap;
    }
    .analyze-btn:hover { background: #333; }
    .analyze-btn:disabled { background: #999; cursor: not-allowed; }

    /* Example pills */
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .pill {
      font-size: 12px;
      color: #555;
      background: #f0f0f0;
      border: 1px solid #e0e0e0;
      border-radius: 20px;
      padding: 4px 12px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .pill:hover { background: #e0e0e0; }

    /* Loading */
    .loading {
      text-align: center;
      padding: 2rem;
      color: #888;
      font-size: 14px;
      display: none;
    }
    .spinner {
      display: inline-block;
      width: 18px; height: 18px;
      border: 2px solid #ddd;
      border-top-color: #555;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin-right: 8px;
      vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Placeholder */
    .placeholder {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 2.5rem;
      text-align: center;
      color: #aaa;
      font-size: 14px;
    }
    .placeholder-icon { font-size: 32px; margin-bottom: 10px; }

    /* Result area */
    #result-area { display: none; }

    /* Winner card */
    .winner-card {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .winner-emoji { font-size: 38px; min-width: 48px; text-align: center; }
    .winner-info { flex: 1; }
    .winner-sublabel {
      font-size: 11px;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 3px;
    }
    .winner-name {
      font-size: 24px;
      font-weight: 600;
      color: #1a1a1a;
    }
    .winner-conf-block { text-align: right; }
    .winner-conf-label {
      font-size: 11px;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 3px;
    }
    .winner-conf-val {
      font-size: 24px;
      font-weight: 600;
    }

    /* Section title */
    .section-title {
      font-size: 12px;
      font-weight: 500;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }

    /* Top 3 */
    .top3 { display: flex; flex-direction: column; gap: 8px; margin-bottom: 1rem; }
    .top3-item {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 10px;
      padding: 12px 16px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .rank-badge {
      width: 28px; height: 28px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 600;
      min-width: 28px;
    }
    .top3-emoji { font-size: 18px; min-width: 24px; }
    .top3-body { flex: 1; }
    .top3-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 5px;
    }
    .top3-name { font-size: 14px; font-weight: 500; color: #1a1a1a; }
    .top3-pct  { font-size: 14px; font-weight: 500; }
    .bar-track {
      background: #f0f0f0;
      border-radius: 4px;
      height: 6px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.7s cubic-bezier(.4,0,.2,1);
    }

    /* All probs */
    .all-probs {
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 12px;
      padding: 1rem 1.25rem;
    }
    .prob-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 4px 0;
    }
    .prob-name {
      font-size: 13px;
      color: #666;
      min-width: 64px;
      text-transform: capitalize;
    }
    .prob-track {
      flex: 1;
      background: #f0f0f0;
      border-radius: 4px;
      height: 5px;
      overflow: hidden;
    }
    .prob-fill {
      height: 100%;
      border-radius: 4px;
    }
    .prob-val {
      font-size: 13px;
      color: #888;
      min-width: 40px;
      text-align: right;
    }

    /* Footer */
    .footer {
      margin-top: 2rem;
      text-align: center;
      font-size: 12px;
      color: #bbb;
    }
    .footer a { color: #888; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div class="badge">DistilBERT · 92.60% accuracy</div>
    <h1>Emotion Detection API</h1>
    <p>Detect emotions from tweets and short texts in real time</p>
  </div>

  <!-- Input -->
  <div class="input-card">
    <p class="input-label">Enter a tweet or sentence</p>
    <div class="input-row">
      <input type="text" id="tweet-input"
             placeholder="e.g. I am so excited about this!"
             onkeydown="if(event.key==='Enter') analyze()" />
      <button class="analyze-btn" id="analyze-btn" onclick="analyze()">
        Analyze &rarr;
      </button>
    </div>
    <div class="examples">
      <span class="pill" onclick="setExample('I am so happy today!')">I am so happy today!</span>
      <span class="pill" onclick="setExample('I feel so lonely and sad')">I feel so lonely and sad</span>
      <span class="pill" onclick="setExample('This makes me so angry!')">This makes me so angry!</span>
      <span class="pill" onclick="setExample('I am terrified of what might happen')">I am terrified</span>
      <span class="pill" onclick="setExample('I love you so much')">I love you so much</span>
      <span class="pill" onclick="setExample('Wow, I did not expect that at all!')">Wow, I did not expect that!</span>
    </div>
  </div>

  <!-- Loading -->
  <div class="loading" id="loading">
    <span class="spinner"></span> Analyzing with DistilBERT...
  </div>

  <!-- Placeholder -->
  <div class="placeholder" id="placeholder">
    <div class="placeholder-icon">💬</div>
    Type a sentence above and click Analyze
  </div>

  <!-- Results -->
  <div id="result-area">

    <!-- Winner -->
    <div class="winner-card" id="winner-card">
      <div class="winner-emoji" id="winner-emoji"></div>
      <div class="winner-info">
        <div class="winner-sublabel">Detected emotion</div>
        <div class="winner-name" id="winner-name"></div>
      </div>
      <div class="winner-conf-block">
        <div class="winner-conf-label">Confidence</div>
        <div class="winner-conf-val" id="winner-conf"></div>
      </div>
    </div>

    <!-- Top 3 -->
    <div class="section-title">Top 3 predictions</div>
    <div class="top3" id="top3"></div>

    <!-- All probabilities -->
    <div class="all-probs">
      <div class="section-title" style="margin-bottom:10px;">All probabilities</div>
      <div id="all-bars"></div>
    </div>

  </div>

  <div class="footer">
    Fine-tuned DistilBERT &middot; 6 emotions &middot;
    <a href="/docs">API docs</a> &middot;
    <a href="/redoc">ReDoc</a>
  </div>

</div>

<script>
  const EMOTIONS = {
    sadness:  { emoji: "😢", color: "#185FA5", bg: "#E6F1FB", rankBg: "#dceefb", rankColor: "#185FA5" },
    joy:      { emoji: "😊", color: "#3B6D11", bg: "#EAF3DE", rankBg: "#d8f0c0", rankColor: "#3B6D11" },
    love:     { emoji: "❤️", color: "#993556", bg: "#FBEAF0", rankBg: "#fad5e5", rankColor: "#993556" },
    anger:    { emoji: "😠", color: "#A32D2D", bg: "#FCEBEB", rankBg: "#fad0d0", rankColor: "#A32D2D" },
    fear:     { emoji: "😨", color: "#854F0B", bg: "#FAEEDA", rankBg: "#fce4b4", rankColor: "#854F0B" },
    surprise: { emoji: "😲", color: "#534AB7", bg: "#EEEDFE", rankBg: "#dddcfa", rankColor: "#534AB7" }
  };

  function setExample(text) {
    document.getElementById('tweet-input').value = text;
    analyze();
  }

  async function analyze() {
    const text = document.getElementById('tweet-input').value.trim();
    if (!text) return;

    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    document.getElementById('placeholder').style.display  = 'none';
    document.getElementById('result-area').style.display  = 'none';
    document.getElementById('loading').style.display      = 'block';

    try {
      const res  = await fetch('/predict', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ text })
      });
      const data = await res.json();
      renderResults(data);
    } catch(e) {
      document.getElementById('loading').style.display = 'none';
      document.getElementById('placeholder').style.display = 'block';
      document.getElementById('placeholder').innerHTML =
        '<div class="placeholder-icon">⚠️</div>Connection error. Is the API running?';
    } finally {
      btn.disabled = false;
      btn.innerHTML = 'Analyze &rarr;';
    }
  }

  function renderResults(data) {
    document.getElementById('loading').style.display     = 'none';
    document.getElementById('result-area').style.display = 'block';

    const em = EMOTIONS[data.emotion];

    // Winner
    document.getElementById('winner-emoji').textContent = em.emoji;
    document.getElementById('winner-name').textContent  =
      data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
    document.getElementById('winner-conf').textContent  = data.confidence.toFixed(1) + '%';
    document.getElementById('winner-conf').style.color  = em.color;
    document.getElementById('winner-card').style.borderLeft = '3px solid ' + em.color;

    // Sort all probs
    const sorted = Object.entries(data.all_probabilities)
                         .sort((a,b) => b[1] - a[1]);

    // Top 3
    const top3el = document.getElementById('top3');
    top3el.innerHTML = '';
    sorted.slice(0,3).forEach(([emotion, pct], i) => {
      const e = EMOTIONS[emotion];
      top3el.innerHTML += `
        <div class="top3-item">
          <div class="rank-badge"
               style="background:${e.rankBg};color:${e.rankColor};">${i+1}</div>
          <div class="top3-emoji">${e.emoji}</div>
          <div class="top3-body">
            <div class="top3-header">
              <span class="top3-name">
                ${emotion.charAt(0).toUpperCase()+emotion.slice(1)}
              </span>
              <span class="top3-pct" style="color:${e.color};">
                ${pct.toFixed(1)}%
              </span>
            </div>
            <div class="bar-track">
              <div class="bar-fill"
                   style="width:${pct.toFixed(1)}%;background:${e.color};"></div>
            </div>
          </div>
        </div>`;
    });

    // All bars
    const allEl = document.getElementById('all-bars');
    allEl.innerHTML = '';
    sorted.forEach(([emotion, pct]) => {
      const e = EMOTIONS[emotion];
      allEl.innerHTML += `
        <div class="prob-row">
          <span class="prob-name">${emotion}</span>
          <div class="prob-track">
            <div class="prob-fill"
                 style="width:${pct.toFixed(1)}%;background:${e.color};"></div>
          </div>
          <span class="prob-val">${pct.toFixed(1)}%</span>
        </div>`;
    });
  }
</script>
</body>
</html>
"""

# ── Endpoints ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def interface():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "healthy", "model": "DistilBERT", "accuracy": "92.60%"}

@app.post("/predict")
def predict(input: TextInput):
    try:
        result = predict_emotion(input.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def batch_predict(input: BatchInput):
    try:
        results = predict_batch(input.texts)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)