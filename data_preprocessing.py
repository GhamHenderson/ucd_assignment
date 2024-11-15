import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Use Chatgpt to generate a function to clean the tweet text for me based on findings from the data understanding.
def clean_tweet_text(tweet):
    # Convert tweet to lowercase for uniformity
    tweet = tweet.lower()

    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)

    # Remove mentions (@username) and hashtags (#hashtag), preserving the words for context if needed
    tweet = re.sub(r'@\w+', '', tweet)  # Removes mentions entirely
    tweet = re.sub(r'#(\w+)', r'\1', tweet)  # Removes only the hashtag symbol, keeping the word

    # Remove special characters, punctuations (keeping meaningful text characters)
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Keeps letters, numbers, and spaces

    # Remove digits, if not needed (alternatively, keep them if relevant)
    tweet = re.sub(r'\d+', '', tweet)

    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    return tweet

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_tweet_text(df_train, df_test):
    # Clean and preprocess the training data
    df_train['cleaned_text'] = df_train['text'].apply(clean_tweet_text)
    df_test['cleaned_text'] = df_test['text'].apply(clean_tweet_text)

    # Drop empty rows
    df_train = df_train.dropna(subset=['cleaned_text'])
    df_test = df_test.dropna(subset=['cleaned_text'])

    # Remove duplicates
    df_train = df_train.drop_duplicates(subset=['cleaned_text'])
    df_test = df_test.drop_duplicates(subset=['cleaned_text'])

    return df_train, df_test

def visualise_findings(df_train, df_test):

    print(df_train.head())
    # 1. Plot the distribution of tweet lengths before and after cleaning
    df_train['original_length'] = df_train['text'].apply(len)
    df_train['cleaned_length'] = df_train['cleaned_text'].apply(len)

    plt.figure(figsize=(14, 6))
    sns.histplot(df_train['original_length'], color='blue', kde=True, label="Original")
    sns.histplot(df_train['cleaned_length'], color='green', kde=True, label="Cleaned")
    plt.title('Distribution of Tweet Lengths Before and After Cleaning')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # 2. Plot the top 20 most common words in the cleaned data

    # Ensure stopwords are downloaded
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Tokenize and filter out stopwords
    all_words = ' '.join(df_train['cleaned_text']).split()
    filtered_words = [word for word in all_words if word not in stop_words]
    word_counts = Counter(filtered_words)

    # Get the 20 most common words
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)

    plt.figure(figsize=(14, 6))
    sns.barplot(x=list(words), y=list(counts), palette="viridis")
    plt.title('Top 20 Most Common Words in Cleaned Tweets')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    # 3. Compare tweet lengths between train and test data after cleaning
    df_test['cleaned_length'] = df_test['cleaned_text'].apply(len)

    plt.figure(figsize=(14, 6))
    sns.histplot(df_train['cleaned_length'], color='blue', kde=True, label="Train")
    sns.histplot(df_test['cleaned_length'], color='orange', kde=True, label="Test")
    plt.title('Comparison of Tweet Lengths in Train and Test Data After Cleaning')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # 4. Compare the distribution of labels in the training and test data
    plt.figure(figsize=(14, 6))
    sns.countplot(x='labels', data=df_train, palette='viridis')
    plt.title('Distribution of Labels in Training Data')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.show()

    # check for outliers in the data by plotting a boxplot
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='labels', y='cleaned_length', data=df_train, palette='viridis')
    plt.title('Boxplot of Tweet Lengths by Labels in Training Data')
    plt.xlabel('Labels')
    plt.ylabel('Tweet Length (characters)')
    plt.show()
    # OUTLIER WAS FOUND WILL NEED TO GO BACK AND MODIFY THE CLEANING FUNCTION

    # Scatter plot (Sentiment vs. Tweet Length)
    plt.figure(figsize=(14, 6))
    plt.scatter(df_train['cleaned_length'], df_train['labels'])
    plt.title('Sentiment Labels vs. Tweet Length')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Labels')
    plt.show()

    # Word cloud
    text = ' '.join(df_train['cleaned_text'])
    wordcloud = WordCloud(width=800, height=600).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Cleaned Tweets')
    plt.show()

    # Bar Plot
    sns.countplot(x='labels', data=df_train)
    plt.title('Count of Sentiment Labels')
    plt.show()

    # Pie Chart
    plt.pie(df_train['labels'].value_counts(), labels=df_train['labels'].value_counts().index, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.show()

    # Correlation Matrix
    corr_matrix = df_train[[all_words, 'sentiment_strength']].corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.show()
  # Violin plot (Tweet Length by Labels)
    plt.figure(figsize=(14, 6))
    sns.violinplot(x='labels', y='cleaned_length', data=df_train, showmeans=True, palette='Set2')
    plt.title('Violin Plot of Tweet Lengths by Labels')
    plt.xlabel('Labels')
    plt.ylabel('Tweet Length (characters)')
    plt.show()

    # Contingency Table
    contingency_table = pd.crosstab(df_train['labels'], df_train['Developer'])

    # Heatmap
    sns.heatmap(contingency_table, annot=True)
    plt.title('Contingency Table as Heatmap')
    plt.show()

    # Count plot to visualize sentiment distribution by developer
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Developer', hue='labels', data=df_train)
    plt.title('Sentiment Distribution by Developer')
    plt.xlabel('Developer')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.show()