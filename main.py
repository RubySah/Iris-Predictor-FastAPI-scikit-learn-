"""FastAPI app that serves the saved iris model through a simple HTML form.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import pickle
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"
templates = Jinja2Templates(directory=BASE_DIR / "templates")

bundle = {}  # holds the loaded model once the app starts


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Run `python train_model.py` first.")
    with open(MODEL_PATH, "rb") as f:
        bundle.update(pickle.load(f))
    yield
    bundle.clear()


app = FastAPI(title="Iris Predictor", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None, "values": {}})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(..., ge=0, le=20),
    sepal_width: float = Form(..., ge=0, le=20),
    petal_length: float = Form(..., ge=0, le=20),
    petal_width: float = Form(..., ge=0, le=20),
):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    model = bundle["model"]

    probabilities = model.predict_proba(features)[0]
    class_index = int(probabilities.argmax())  # position in target_names

    result = {
        "species": bundle["target_names"][class_index],
        "confidence": round(float(probabilities[class_index]) * 100, 1),
        "all": [
            (name, round(float(p) * 100, 1))
            for name, p in zip(bundle["target_names"], probabilities)
        ],
    }
    values = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    return templates.TemplateResponse(request, "index.html", {"result": result, "values": values})
