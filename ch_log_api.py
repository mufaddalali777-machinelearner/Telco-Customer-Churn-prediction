from fastapi import FastAPI, Path
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = FastAPI()

class ModelInput(BaseModel):
    features : list[float] = Field(..., min_length=27, max_length=27)

model1 = joblib.load('classification_model.joblib')
scalar1 = joblib.load('scaler.joblib')

@app.post("/predict-churn-by-classy-model")
def churn_predictor(data: ModelInput):
    input = np.array([data.features])
    input_sc = scalar1.transform(input)
    
    prediction = model1.predict(input_sc)
    return int(prediction[0])