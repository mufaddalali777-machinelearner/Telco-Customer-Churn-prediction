from fastapi import FastAPI, Path
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = FastAPI()

class ModelInput(BaseModel):
    features : list[float] = Field(..., min_length=27, max_length=27)


model3 = joblib.load('dt_model.joblib')




@app.post("/predict-churn-with-trees")
def churn_predictor_xg(data: ModelInput):
    input_data = np.array([data.features])
    prediction = model3.predict(input_data)
        
    # Get probabilities for [Class 0, Class 1]
    probabilities = model3.predict_proba(input_data)
    churn_risk = float(probabilities[0, 1])
        
    return {
        "model": "XGBoost / Decision Tree",
        "churn_prediction": int(prediction[0]),
        "churn_probability_risk": round(churn_risk, 4)
    }
    
