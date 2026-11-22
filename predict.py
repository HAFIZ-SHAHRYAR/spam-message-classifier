from pathlib import Path
import sys
import joblib


MODEL_PATH = Path(__file__).resolve().parent / "spam_classifier.joblib"


def predict(message: str) -> str:
    if not message.strip():
        raise ValueError("Message cannot be empty")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found. Run 'python {Path(__file__).name.replace('predict.py', 'train.py')}' first."
        )
    model = joblib.load(MODEL_PATH)
    return str(model.predict([message])[0])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python predict.py "message to classify"')
    print(predict(" ".join(sys.argv[1:])))
