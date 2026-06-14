from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
import joblib

app = FastAPI(title="Neural Network Churn Service")

# 1. Define the input schema matching your 27 customer features
class ChurnInput(BaseModel):
    features: list[float] = Field(..., min_length=27, max_length=27)

# 2. Load the ONNX runtime inference session instead of TensorFlow
# Also load your scaler if your neural network needs normalized inputs
try:
    session = ort.InferenceSession("churn_nn_model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    scaler = joblib.load("scaler.joblib") # Remove if you don't scale inputs for the NN
except Exception as e:
    print(f"Error loading model artifacts: {e}")

@app.post("/predict")
def predict_nn(data: ChurnInput):
    try:
        # 3. Convert input features to a 2D float32 numpy array
        input_data = np.array([data.features], dtype=np.float32)
        
        # 4. Apply scaling if your NN was trained on scaled data
        if 'scaler' in globals():
            input_data = scaler.transform(input_data).astype(np.float32)
        
        # 5. Run inference through the ONNX runtime session
        raw_prediction = session.run([output_name], {input_name: input_data})[0]
        
        # Extract the probability score from the prediction tensor
        probability = float(raw_prediction[0][0])
        binary_prediction = 1 if probability >= 0.5 else 0
        
        return {
            "model": "Neural Network (ONNX Optimized)",
            "churn_prediction": binary_prediction,
            "churn_probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))