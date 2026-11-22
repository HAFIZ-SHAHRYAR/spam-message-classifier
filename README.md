# Spam Message Classifier

A beginner-friendly machine-learning project that uses TF-IDF text features and a Linear Support Vector Machine to classify text messages as `spam` or `ham` (not spam).

## Setup

```bash
cd spam_classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py
```

This prints an accuracy report, creates `spam_classifier.joblib`, and saves the confusion matrix to `confusion_matrix.png`.

## Make predictions

```bash
python predict.py "Congratulations! You won a free gift"
python predict.py "Are we still meeting at 5?"
```

## Explore in Jupyter

```bash
jupyter notebook spam_classifier.ipynb
```

## Dataset

The small educational dataset in `data/messages.csv` contains example messages. Replace it with a larger, ethically sourced dataset for real use.

## Project structure

- `train.py`: trains and evaluates the model
- `predict.py`: loads the trained model and classifies messages
- `spam_classifier.ipynb`: beginner-friendly data exploration and training notebook
- `data/messages.csv`: example training data
- `tests/test_classifier.py`: basic tests
