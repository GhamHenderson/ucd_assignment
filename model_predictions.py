import numpy as np
from keras.models import load_model
from data_preprocessing import clean_tweet_text

model = load_model("model.keras")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_sentiment(prediction):
    # Get the index of the maximum value in the prediction array
    sentiment_index = np.argmax(prediction)

    # Define the sentiment labels
    sentiment_labels = [
        "Very Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very Positive",
    ]

    # Get the corresponding sentiment label
    sentiment = sentiment_labels[sentiment_index]

    return sentiment


def get_sentiment_message(prediction):
    # Get the sentiment label
    sentiment = get_sentiment(prediction)

    if sentiment == "Very Negative":
        message = "Oh no! It seems like this sentiment is very negative."
    elif sentiment == "Negative":
        message = "It seems somewhat negative."
    elif sentiment == "Neutral":
        message = "This sentiment appears neutral."
    elif sentiment == "Positive":
        message = "This seems somewhat positive!"
    elif sentiment == "Very Positive":
        message = "Fantastic! This is very positive."

    return message


# Example setup:
MAX_SEQUENCE_LENGTH = 100  # Define your desired sequence length


def predict_sentiment(text):
    # Clean and preprocess the text
    cleaned_text = clean_tweet_text(text)
    print("Cleaned text:", cleaned_text)

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned_text])
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    print("Tokenized text:", tokenized_text)

    # Pad the tokenized text to ensure consistent input length
    padded_text = pad_sequences(
        tokenized_text, maxlen=MAX_SEQUENCE_LENGTH, padding="post"
    )

    # Make the prediction
    prediction = model.predict(padded_text)
    print("Prediction probabilities:", prediction)

    return prediction


# Example usage:
very_negative = "I am extremely unhappy with this game, it is so bad, this is a very negative sentiment"
negative = "This game is bad I really dont like it,  this is a negative sentiment"
neutral = "very average game not good not bad, this is a neutral sentiment"
positive = "I am happy with this game, this is a positive day"

text = input("Enter the text for the model to predict: ")

prediction = predict_sentiment(text=text)


message = get_sentiment_message(prediction)

print("\n\n PREDICTION: ", message)
