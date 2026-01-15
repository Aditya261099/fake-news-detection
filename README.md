# Fake News Detection with TensorFlow

This project implements a fake news detection model using TensorFlow and pre-trained GloVe word embeddings. It classifies news headlines as real or fake based on the news title. [page:1]

## Dataset

- Source: Fake news dataset from GeeksforGeeks (news.csv: title, text, label = REAL/FAKE). [page:1]
- Labels are encoded to numeric values using `LabelEncoder`. [page:1]

## Model

- Embedding layer initialized with GloVe 50d vectors (glove.6B.50d.txt). [page:1]
- Architecture: Embedding → Dropout → Conv1D → MaxPooling1D → LSTM(64) → Dense(1, sigmoid). [page:1]
- Trained for 50 epochs on 3,000 samples (title only), with a 10% validation split. [page:1]
- Observed performance: ~98% training accuracy and ~74–77% validation accuracy, with some overfitting. [page:1]

## Project Structure

- `fake_news_detector.py` – Loads data, preprocesses, builds, trains, and saves the model.
- `requirements.txt` – Python dependencies.
- `models/fake_news_lstm.h5` – Trained model (ignored in git).
- `data/news.csv` – Dataset (ignored in git).

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Put news.csv under data/ and glove.6B.50d.txt in project root
python fake_news_detector.py


