import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70

CONFIG_KEYS = {"model_type", "model_params", "train_data_paths", "eval_path"}


def _load_training_data(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _build_model(model_type: str, params: dict):
    if model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=42)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(**params, random_state=42)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _get_model_params(params: dict) -> dict:
    if "model_params" in params:
        return params["model_params"]
    return {key: value for key, value in params.items() if key not in CONFIG_KEYS}


def train(
    params: dict,
    data_path: str | None = None,
    eval_path: str | None = None,
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier.
        data_path: đường dẫn đến file dữ liệu huấn luyện.
        eval_path: đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """

    model_type = params.get("model_type", "random_forest")
    model_params = _get_model_params(params)
    train_paths = params.get("train_data_paths", ["data/train_phase1.csv"])
    if data_path is not None:
        train_paths = [data_path]
    eval_path = eval_path or params.get("eval_path", "data/eval.csv")

    df_train = _load_training_data(train_paths)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        model = _build_model(model_type, model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train(params)
