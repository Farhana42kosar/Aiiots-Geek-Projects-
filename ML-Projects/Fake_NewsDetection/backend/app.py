from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ---------- Load Model ----------
model = joblib.load("model/fake_news_pipeline.pkl")

# ---------- FastAPI App ----------
app = FastAPI(
    title="Fake News Detection API",
    description="Predict whether a news article is Fake or Real",
    version="1.0"
)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Static & Templates ----------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

# ---------- Schemas ----------
class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    label: int
    confidence: float

# ---------- Frontend Route ----------
@app.get("/")
def frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Prediction API ----------
@app.post("/predict", response_model=NewsResponse)
def predict_news(data: NewsRequest):
    pred = model.predict([data.text])[0]
    prob = model.predict_proba([data.text]).max()

    return {
        "prediction": "REAL NEWS" if pred == 1 else "FAKE NEWS",
        "label": int(pred),
        "confidence": round(float(prob), 3)
    }
