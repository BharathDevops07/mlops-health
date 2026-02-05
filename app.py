from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class InputData(BaseModel):
    text: str

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Health ML API running"}

@app.post("/predict")
def predict(data: InputData):
    X = vectorizer.transform([data.text])
    pred = model.predict(X)[0]
    return {"prediction": pred}
