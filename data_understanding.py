import pandas as pd
from collections import Counter

def data_understanding():
    df_train = pd.read_csv('training/twitter_training.csv')
    df_test = pd.read_csv("validation/twitter_validation.csv")

    # make all rows show up
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.width',
                  None)  # Adjust display width as needed
    pd.set_option('display.max_colwidth', None)  # Display full column width


    print(df_train.head())
    print(df_test.head())

    print(df_train.describe())
    print(df_test.describe())

    # I See this data has no header labels, so I will add them
    df_test.columns = ['Header', 'Developer','labels','text']
    df_train.columns = ['Header', 'Developer','labels','text']

    # confirm the changes
    print("\n", df_train.head())
    print("\n",df_test.head())

    # Analyze the data shape, info, description, duplicates, and missing values
    print("\n",df_train.shape)
    print("\n",df_train.info())
    print("\n",df_train.describe())
    print("\n",df_train.duplicated().sum())
    print("\n",df_train.isnull().sum())
    # Shape: (74681, 4),

    print("\n",df_test.shape)
    print("\n",df_test.info())
    print("\n",df_test.describe())
    print("\n",df_test.duplicated().sum())
    print("\n",df_test.isnull().sum())
    # Shape: (999, 4)

    return df_train, df_test

