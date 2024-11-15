def data_cleaning(df_train, df_test):
    # Drop duplicates
    df_train.drop_duplicates(inplace=True)
    df_test.drop_duplicates(inplace=True)

    # Drop missing values
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    # Confirm the changes
    print("\n", df_train.isnull().sum())
    print("\n", df_test.isnull().sum())

    # Drop Undesired Columns
    df_train.drop(columns=['Header', 'Developer'], inplace=True)
    df_test.drop(columns=['Header', 'Developer'], inplace=True)

    # Confirm the changes
    print("\n", df_train.head())
    print("\n", df_test.head())

    df_test_cleaned = df_test.copy()
    df_train_cleaned = df_train.copy()

    # Remove extreme outliers from the training data in terms of tweet length.
    df_train_cleaned = df_train_cleaned[df_train_cleaned['text'].apply(len) < 600]

    return df_train_cleaned, df_test_cleaned
