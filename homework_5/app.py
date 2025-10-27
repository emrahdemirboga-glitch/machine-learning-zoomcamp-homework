import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pre-trained pipeline
with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

# Define request schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# FastAPI app
app = FastAPI(title="Lead Scoring API")

# Prediction endpoint
@app.post("/predict")
def predict(lead: Lead):
    record = [lead.dict()]
    proba = pipeline.predict_proba(record)[0][1]
    return {"converted_proba": round(proba, 3)}

# Optional health check
@app.get("/")
def root():
    return {"status": "API is running"}
