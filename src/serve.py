import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import boto3
except ImportError:  # pragma: no cover - handled in runtime env
    boto3 = None

app = FastAPI()

S3_BUCKET = os.environ.get("CLOUD_BUCKET") or os.environ.get("AWS_BUCKET")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/model.pkl"))
model = None


def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    if S3_BUCKET is None:
        if os.path.exists(MODEL_PATH):
            return
        raise RuntimeError("CLOUD_BUCKET/AWS_BUCKET is not set and no local model exists")

    if boto3 is None:
        raise RuntimeError("boto3 is required to download the model")

    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print("Model downloaded from S3.")


def get_model():
    """Load model lazily so /health can work even before the model is downloaded."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            download_model()
        model = joblib.load(MODEL_PATH)
    return model


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint suy luận."""
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    loaded_model = get_model()
    pred = int(loaded_model.predict([req.features])[0])
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred, "label": labels.get(pred, "unknown")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
