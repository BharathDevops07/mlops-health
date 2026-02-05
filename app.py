from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Health ML API running"}

@app.get("/predict")
def predict(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return {"prediction": pred}
