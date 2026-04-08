# MindMend AI — Emotion Aware Adaptive Learning Tutor

An AI tutor that observes student emotions (webcam + microphone) and adapts teaching style in real time.

## Features

- **Facial emotion detection** — CNN on 48×48 grayscale (FER2013)
- **Voice emotion detection** — LSTM on MFCC features (RAVDESS)
- **Cognitive state fusion** — Maps emotions → confused / frustrated / bored / engaged / stressed
- **Adaptive teaching** — Prompt style changes based on learning state
- **Closed feedback loop** — Affective computing pipeline

## Tech Stack

- **Backend:** Python FastAPI
- **ML:** PyTorch, OpenCV, Librosa
- **Frontend:** HTML + vanilla JS

---

## Dataset Setup

Place datasets in the `data/` folder:

### 1. FER2013 (Facial Emotion)

1. Download from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Create folder: `data/fer2013/`
3. Place `fer2013.csv` in `data/fer2013/fer2013.csv`

### 2. RAVDESS (Voice Emotion)

1. Download from [Kaggle RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
2. Extract to `data/ravdess/`
3. Expected structure: `data/ravdess/Actor_01/`, `Actor_02/`, … or similar with `.wav` files

---

## Installation & Run

### 1. Install dependencies

```bash
cd MindMed
pip install -r requirements.txt
```

### 2. Train models (requires datasets)

**Facial emotion CNN:**
```bash
python -m ml.training.train_facial_emotion --epochs 30 --batch-size 64
```

Model saved to `data/models/emotion_cnn.pth`.

**Voice emotion LSTM:**
```bash
python -m ml.training.train_voice_emotion --epochs 50 --batch-size 32
```

Model saved to `data/models/voice_emotion_lstm.pth`.

**Optional — Cognitive fusion RandomForest:**
```bash
python -m ml.training.train_fusion_rf
```

Saved to `data/models/cognitive_fusion_rf.joblib`.

### 3. Start backend

```bash
uvicorn backend.main:app --reload
```

### 4. Open frontend

Open in browser: **http://localhost:8000/app**

- Enter a topic (e.g. "recursion in Python")
- Click **Ask**
- Allow camera and microphone when prompted (for emotion capture)
- The tutor adapts its response to your inferred learning state

---

## Project Structure

```
mindmend-ai/
├── backend/
│   ├── api/           # FastAPI routes
│   ├── services/      # Webcam, mic, LLM
│   ├── fusion/        # Cognitive state fusion
│   ├── tutor/         # Adaptive prompt generator
│   └── main.py
├── ml/
│   ├── vision/        # FER2013 loader, CNN
│   ├── audio/         # RAVDESS loader, MFCC, LSTM
│   └── training/      # Train scripts
├── frontend/
│   └── index.html
├── data/
│   ├── fer2013/
│   ├── ravdess/
│   ├── models/
│   └── plots/
├── requirements.txt
└── README.md
```

---

## API

- `POST /api/ask` — Body: `{"topic": "your topic"}`  
  Returns: `{ response, learning_state, face_emotion, voice_emotion }`

---

## LLM Integration (Ollama)

The app uses **Ollama** for local LLM inference by default.

**Setup:**
```bash
# Install Ollama from https://ollama.com

# Start Ollama (usually runs in background after install)
ollama serve

# Pull a model (e.g. llama2)
ollama pull llama2
```

**Environment variables** (optional):
- `OLLAMA_BASE_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — default `llama2` (use `llama3`, `mistral`, etc.)
- `OLLAMA_TIMEOUT` — default `60` seconds

If Ollama is not running, the app falls back to a placeholder response.

---

## Notes

- **Webcam & microphone:** The backend captures from the server’s devices. For local use, run backend and open frontend on the same machine.
- **Models:** Without trained models, random weights are used; training is strongly recommended.
- **PyAudio:** On Windows, if `pip install pyaudio` fails, use a prebuilt wheel or `conda install pyaudio`.
