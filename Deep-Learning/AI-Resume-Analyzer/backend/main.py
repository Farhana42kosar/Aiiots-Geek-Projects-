import io
import re
import torch
from fastapi import FastAPI, UploadFile, File
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from processor import get_job_recommendations, calculate_ats_score

app = FastAPI()

# LOAD BERT MODEL

MODEL_PATH = "models" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# LABEL MAPPING (IMPORTANT)
 
id2label = {
    0: 'Accountant', 1: 'Advocate', 2: 'Agriculture', 3: 'Apparel',
    4: 'Architecture', 5: 'Arts', 6: 'Automobile', 7: 'Aviation',
    8: 'BPO', 9: 'Banking', 10: 'Blockchain', 11: 'Building and Construction',
    12: 'Business Analyst', 13: 'Civil Engineer', 14: 'Consultant',
    15: 'Data Science', 16: 'Database', 17: 'Designing', 18: 'DevOps',
    19: 'Digital Media', 20: 'DotNet Developer', 21: 'ETL Developer',
    22: 'Education', 23: 'Electrical Engineering', 24: 'Finance',
    25: 'Food and Beverages', 26: 'Health and Fitness', 27: 'Human Resources',
    28: 'Information Technology', 29: 'Java Developer', 30: 'Management',
    31: 'Mechanical Engineer', 32: 'Network Security Engineer',
    33: 'Operations Manager', 34: 'PMO', 35: 'Public Relations',
    36: 'Python Developer', 37: 'React Developer', 38: 'SAP Developer',
    39: 'SQL Developer', 40: 'Sales', 41: 'Testing', 42: 'Web Designing'
}

# TEXT CLEANING FUNCTION

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# PREDICT ENDPOINT

@app.post("/predict")
async def predict_resume(file: UploadFile = File(...)):
    
    # READ PDF

    content = await file.read()
    pdf = PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    processed_text = clean_text(text)

    
    # BERT PREDICTION
    
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_label = id2label[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item()

    # JOB RECOMMENDATIONS

    recommendations = get_job_recommendations(text, predicted_label)
    top_match_pct = recommendations[0]['match'] if recommendations else 0

    
    # ATS SCORE

    ats_score = calculate_ats_score(text, top_match_pct)

    
    return {
        "filename": file.filename,
        "predicted_domain": predicted_label,
        "ats_score": ats_score,
        "job_recommendations": recommendations
    }
