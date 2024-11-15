import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from data_preprocessing import clean_tweet_text


def data_separation(df_train, df_test):
    # Separate features and labels for training and test data
    train_texts = df_train["text"].values
    train_labels = df_train["labels"].values
    test_texts = df_test["text"].values
    test_labels = df_test["labels"].values

    # Initialize and encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Initialize tokenizer with a set vocabulary size and OOV token
    max_vocab_size = 33829
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    # Convert texts to sequences
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Define maximum sequence length and pad sequences
    maxlen = max(len(seq) for seq in train_sequences)
    train_padded = pad_sequences(train_sequences, maxlen=maxlen, padding="post")
    test_padded = pad_sequences(test_sequences, maxlen=maxlen, padding="post")

    # Set `input_dim` based on the tokenizer's vocabulary size or `max_vocab_size`
    input_dim = min(len(tokenizer.word_index) + 1, max_vocab_size)
    return (
        input_dim,
        maxlen,
        test_labels_encoded,
        test_padded,
        train_labels_encoded,
        train_padded,
    )


def model_training(
    input_dim,
    maxlen,
    test_labels_encoded,
    test_padded,
    train_labels_encoded,
    train_padded,
):
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=maxlen))
    # LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                128,
                kernel_regularizer=l2(0.1),
                return_sequences=True,
                recurrent_regularizer=l2(0.1),
            )
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # LSTM layer
    model.add(
        Bidirectional(
            LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Dense layer
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    # Compile and train
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    history = model.fit(
        train_padded,
        train_labels_encoded,
        validation_data=(test_padded, test_labels_encoded),
        epochs=10,
    )
    test_loss, test_accuracy = model.evaluate(test_padded, test_labels_encoded)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    model.save("model.keras")
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()
