from data_cleaning import data_cleaning
from data_preprocessing import preprocess_tweet_text, visualise_findings
from data_splitting import data_separation, model_training
from data_understanding import data_understanding

# Get the datasets and analyze them
df_train, df_test = data_understanding()

# Clean the datasets of duplicates, missing values, and undesired columns
df_train_clean, df_test_clean = data_cleaning(df_train, df_test)

# Preprocess the tweet text in the datasets for analysis and modeling purposes
df_train_preprocessed, df_test_preprocessed = preprocess_tweet_text(df_train_clean, df_test_clean)

# Visualize the findings of the cleaned and preprocessed datasets
visualise_findings(df_train_preprocessed, df_test_preprocessed)

# # Split the data into training and validation sets
input_dim, maxlen, test_labels_encoded, test_padded, train_labels_encoded, train_padded = data_separation(df_train_preprocessed, df_test_preprocessed)

model = model_training(input_dim, maxlen, test_labels_encoded, test_padded, train_labels_encoded, train_padded)


# Save the cleaned datasets now as CSV files as a checkpoint as I don't want to do this every time I run the code.
df_train_clean.to_csv('training/twitter_training_cleaned.csv', index=False)
df_test_clean.to_csv('validation/twitter_validation_cleaned.csv', index=False)

