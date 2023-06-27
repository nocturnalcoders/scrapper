from flask import Flask, render_template
import pandas as pd
import re
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import seaborn as sns

app = Flask(__name__)

# Set the template folder path
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app.template_folder = template_dir


lexicon_positive = dict()
lexicon_negative = dict()


# Function to load lexicons from CSV files
def load_lexicons():
    global lexicon_positive, lexicon_negative

    with open(
        "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\twitter-scrapper\\lexicon_positive.csv",
        "r",
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            lexicon_positive[row[0]] = int(row[1])

    with open(
        "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\twitter-scrapper\\lexicon_negative.csv",
        "r",
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            lexicon_negative[row[0]] = int(row[1])


@app.route("/preprocessed")
def preprocessed():
    # Baca data tweet dari file CSV
    tweets_data = pd.read_excel(
        "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\dataemoji.xlsx"
    )
    tweets = tweets_data["Tweets"]

    # Inisialisasi objek Stemmer untuk stemming
    stemmer = PorterStemmer()

    # Pra-pemrosesan teks
    preprocessed_tweets = []

    def cleaningText(text):
        text = re.sub(r"@[A-Za-z0-9]+", "", text)  # remove mentions
        text = re.sub(r"#[A-Za-z0-9]+", "", text)  # remove hashtag
        text = re.sub(r"RT[\s]", "", text)  # remove RT
        text = re.sub(r"http\S+", "", text)  # remove link
        text = re.sub(r"[0-9]+", "", text)  # remove numbers

        text = text.replace("\n", " ")  # replace new line into space
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # remove all punctuations
        text = text.strip(" ")  # remove characters space from both left and right text
        return text

    def casefoldingText(text):
        text = text.lower()  # Converting all the characters in a text into lower case
        return text

    def tokenizingText(text):
        text = word_tokenize(
            text
        )  # Tokenizing or splitting a string, text into a list of tokens
        return text

    def filteringText(text):
        listStopwords = set(stopwords.words("indonesian"))  # Remove stopwords in a text
        filtered = []
        for txt in text:
            if txt not in listStopwords:
                filtered.append(txt)
        return filtered

    def stemmingText(text):
        text = [stemmer.stem(word) for word in text]  # Reducing a word to its word stem
        return text

    def toSentence(list_words):
        sentence = " ".join(list_words)  # Convert list of words into sentence
        return sentence

    # Preprocessing tweets data
    tweets_clean = tweets.apply(cleaningText)
    tweets_clean = tweets_clean.apply(casefoldingText)

    tweets_preprocessed = tweets_clean.apply(tokenizingText)
    tweets_preprocessed = tweets_preprocessed.apply(filteringText)
    tweets_preprocessed = tweets_preprocessed.apply(stemmingText)

    preprocessed_data = pd.DataFrame(
        {"preprocessed_tweet": tweets_preprocessed, "tweets": tweets_clean}
    )

    preprocessed_data["preprocessed_tweet"] = preprocessed_data[
        "preprocessed_tweet"
    ].apply(toSentence)

    return preprocessed_data


def sentiment_analysis_lexicon_indonesia(text):
    words = word_tokenize(text)
    score = 0
    for word in words:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]

    if score > 0:
        polarity = "Positif"
    elif score < 0:
        polarity = "Negatif"
    else:
        polarity = "Netral"

    return score, polarity


@app.route("/")
def home():
    preprocessed_data = preprocessed()
    load_lexicons()  # Load lexicons before sentiment analysis

    # Add 'polarity' column to preprocessed_data
    preprocessed_data["polarity"] = preprocessed_data["preprocessed_tweet"].apply(
        sentiment_analysis_lexicon_indonesia
    )

    # Generate chart data
    polarity_counts = preprocessed_data["polarity"].value_counts().to_dict()
    chart_data = {
        "labels": list(polarity_counts.keys()),
        "values": list(polarity_counts.values()),
    }

    # Convert label values to strings
    chart_data["labels"] = [str(label) for label in chart_data["labels"]]

    # Generate bar plot for polarity counts
    plt.figure(figsize=(8, 6))
    sns.barplot(x=chart_data["labels"], y=chart_data["values"])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis")

    # Get the absolute path of the 'web' folder
    web_folder = os.path.join(app.static_folder, "web")

    # Save the plot image in the 'web' folder
    sentiment_analysis_path = os.path.join(web_folder, "sentiment_analysis.png")
    plt.savefig(sentiment_analysis_path)
    plt.close()

    # Generate Word Cloud
    text = " ".join(preprocessed_data["preprocessed_tweet"])
    wordcloud = WordCloud(width=800, height=400).generate(text)

    # Save Word Cloud as image
    wordcloud_path = os.path.join(web_folder, "wordcloud.png")
    wordcloud.to_file(wordcloud_path)

    return render_template("visualization.html")


def naive_bayes_classification(preprocessed_data):
    X = preprocessed_data["preprocessed_tweet"]
    y = preprocessed_data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_vectorized, y_train)

    y_pred = naive_bayes.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


@app.route("/sentiment")
def sentiment():
    preprocessed_data = preprocessed()
    load_lexicons()  # Load lexicons before sentiment analysis

    # Add 'sentiment_score' and 'polarity' columns to preprocessed_data
    preprocessed_data["sentiment_score"], preprocessed_data["polarity"] = zip(
        *preprocessed_data["preprocessed_tweet"].apply(
            sentiment_analysis_lexicon_indonesia
        )
    )

    # Separate 'polarity' into 'score' and 'label' columns
    preprocessed_data["score"], preprocessed_data["label"] = zip(
        *preprocessed_data["polarity"]
    )

    # Calculate polarity counts
    polarity_counts = preprocessed_data["label"].value_counts().to_dict()

    # Calculate total sentiment score
    total_sentiment_score = preprocessed_data["sentiment_score"].sum()

    print("Total Sentiment Score:", total_sentiment_score)

    return render_template("sentiment.html", polarity_counts=polarity_counts)


if __name__ == "__main__":
    preprocessed_data = preprocessed()

    load_lexicons()  # Load lexicons before sentiment analysis

    # Add 'polarity' column to preprocessed_data
    preprocessed_data["polarity"] = preprocessed_data["preprocessed_tweet"].apply(
        sentiment_analysis_lexicon_indonesia
    )

    # Separate 'polarity' into 'score' and 'label' columns
    preprocessed_data[["score", "label"]] = preprocessed_data["polarity"].apply(
        lambda x: pd.Series(x)
    )

    print("Sentiment Analysis using Lexicon:")
    print(
        preprocessed_data
    )  # Output preprocessed data with polarity, score, and label columns

    accuracy = naive_bayes_classification(preprocessed_data)
    print("Accuracy:", accuracy)
    app.run(debug=False)
