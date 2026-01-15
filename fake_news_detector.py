import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load dataset
data = pd.read_csv("data/news.csv")

#data = pd.read_csv("news.csv")
data = data.drop(["Unnamed: 0"], axis=1)

# 2. Encode labels (REAL/FAKE -> 0/1)
le = preprocessing.LabelEncoder()
le.fit(data["label"])
data["label"] = le.transform(data["label"])

# 3. Hyperparameters / config
embedding_dim = 50
max_length = 54
padding_type = "post"
trunc_type = "post"
oov_tok = "<OOV>"
training_size = 3000
test_portion = 0.1

# 4. Prepare text and labels (titles only, first 3000 rows)
titles = []
texts = []
labels = []

for i in range(training_size):
    titles.append(data["title"][i])
    texts.append(data["text"][i])
    labels.append(data["label"][i])

labels = np.array(labels)

# 5. Tokenize titles and pad sequences
tokenizer1 = Tokenizer(oov_token=oov_tok)
tokenizer1.fit_on_texts(titles)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)

sequences1 = tokenizer1.texts_to_sequences(titles)
padded1 = pad_sequences(
    sequences1,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type,
)

# 6. Train / test split
split = int(test_portion * training_size)

test_sequences1 = padded1[0:split]
test_labels = labels[0:split]

training_sequences1 = padded1[split:training_size]
training_labels = labels[split:training_size]

training_sequences1 = np.array(training_sequences1)
test_sequences1 = np.array(test_sequences1)

# 7. Load GloVe embeddings (assumes glove.6B.50d.txt is in the same folder)
embedding_index = {}
with open("glove.6B.50d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size1 + 1, embedding_dim))
for word, i in word_index1.items():
    if i < vocab_size1:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 8. Build model (Embedding -> Dropout -> Conv1D -> MaxPool -> LSTM -> Dense)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            vocab_size1 + 1,
            embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            trainable=False,
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

# 9. Train model
history = model.fit(
    training_sequences1,
    training_labels,
    epochs=50,
    validation_data=(test_sequences1, test_labels),
    verbose=2,
)

# 10. Sample prediction
sample_text = "Karry to go to France in gesture of sympathy"
sample_seq = tokenizer1.texts_to_sequences([sample_text])
sample_pad = pad_sequences(
    sample_seq,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type,
)

pred = model.predict(sample_pad, verbose=0)[0][0]
if pred >= 0.5:
    print("This news is True")
else:
    print("This news is False")


model.save("models/fake_news_lstm.h5")
print("Model saved to models/fake_news_lstm.h5")

