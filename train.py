from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "messages.csv"
MODEL_PATH = BASE_DIR / "spam_classifier.joblib"


def load_data(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(path)
    if not {"message", "label"}.issubset(frame.columns):
        raise ValueError("CSV must contain 'message' and 'label' columns")
    frame = frame.dropna(subset=["message", "label"])
    return frame["message"].astype(str).tolist(), frame["label"].astype(str).tolist()


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    messages, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.name}")

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
