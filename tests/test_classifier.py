from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import build_model, load_data


def test_dataset_has_valid_labels():
    messages, labels = load_data()
    assert len(messages) == len(labels)
    assert set(labels) == {"ham", "spam"}


def test_model_trains_and_predicts():
    messages, labels = load_data()
    model = build_model()
    model.fit(messages, labels)
    prediction = model.predict(["Are we still meeting for lunch?"])[0]
    assert prediction in {"ham", "spam"}
