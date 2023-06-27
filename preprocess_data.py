import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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


def casefoldingText(text):  # Converting all the characters in a text into lower case
    text = text.lower()
    return text


def tokenizingText(
    text,
):  # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text


def filteringText(text):  # Remove stopwors in a text
    listStopwords = set(stopwords.words("indonesian"))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text


def stemmingText(
    text,
):  # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text


def toSentence(list_words):  # Convert list of words into sentence
    sentence = " ".join(word for word in list_words)
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

# Tambahkan kolom 'tweets' yang belum dipreproses ke dalam preprocessed_data
preprocessed_data = preprocessed_data.reindex(columns=["preprocessed_tweet", "tweets"])

# Output hasil pra-pemrosesan teks
print(tweets_data)
print(preprocessed_data)

# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv

with open(
    "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\twitter-scrapper\\lexicon_positive.csv",
    "r",
) as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv

with open(
    "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\twitter-scrapper\\lexicon_negative.csv",
    "r",
) as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])


# Function to determine sentiment polarity of tweets
def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]
    if score > 0:
        polarity = "positive"
    elif score < 0:
        polarity = "negative"
    else:
        polarity = "neutral"
    return score, polarity


# Results from determining sentiment polarity of tweets
results = preprocessed_data["preprocessed_tweet"].apply(
    sentiment_analysis_lexicon_indonesia
)
results = list(zip(*results))
preprocessed_data["polarity_score"] = results[0]
preprocessed_data["polarity"] = results[1]
print(preprocessed_data["polarity"].value_counts())

# Export to Excel file
preprocessed_data.to_excel(
    r"C:\Users\DevOps\Documents\Project Cuan\Reza Tanjung Thesis\twitter-scrapper\tweets_data_clean_polarity.xlsx",
    index=False,
    header=True,
    index_label=None,
)

# Output the preprocessed data with polarity information
print(preprocessed_data)
