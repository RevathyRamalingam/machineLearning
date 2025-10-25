import pickle
from fastapi import FastAPI
from typing import Dict, Any
import uvicorn

app=FastAPI(title="lead-convertion")
with open('pipeline_v1.bin','rb') as f_in:
    pipeline = pickle.load(f_in)

lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

@app.post("/convert")
def predict(lead: Dict[str, Any]):
    converted_prob = predict_lead(lead)
    return{
        "probability of getting converted = ":converted_prob,
        "isConverted = ":bool(converted_prob >=0.5)
    }

def predict_lead(lead):
    result=pipeline.predict_proba(lead)[0,1]
    return float(result)

if __name__== "__main__":
    uvicorn.run(app,host="localhost",port=9696)