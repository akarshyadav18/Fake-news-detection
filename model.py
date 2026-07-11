# ============================================================
# FILE: model.py
# PURPOSE: Load data, train the ML model, save it to disk
# ============================================================

# --- IMPORTS ---
# pandas: reads our CSV file into a table we can work with
import pandas as pd

# re: lets us use Regular Expressions to clean text
import re

# nltk: Natural Language Toolkit — gives us stopwords and stemmer
import nltk

# These two download required data files from nltk servers
# 'stopwords' = list of common useless words (the, is, a, an...)
# 'punkt' = sentence/word tokenizer rules
nltk.download('stopwords')
nltk.download('punkt')

# PorterStemmer: reduces words to root form (running → run)
from nltk.stem import PorterStemmer

# stopwords: pre-built list of common English words to remove
from nltk.corpus import stopwords

# TfidfVectorizer: converts text into numbers using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression: our ML model that learns fake vs real
from sklearn.linear_model import LogisticRegression

# train_test_split: splits data into training set and testing set
from sklearn.model_selection import train_test_split

# accuracy_score etc: tools to measure how good our model is
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# pickle: saves our trained model to a file so Flask can use it
import pickle

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================

# Read the CSV file into a DataFrame (think of it as an Excel table)
df = pd.read_csv('news.csv')

# Print first look at the data
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nLabel distribution:")
print(df['label'].value_counts())

# ============================================================
# STEP 2: PREPARE THE DATA
# ============================================================

# We combine 'title' and 'text' columns into one column called 'content'
# Why? Because both the headline and body contain signals for fake/real
# The + " " + joins them with a space between
df['content'] = df['title'] + " " + df['text']

# Check for missing values and remove those rows
# dropna() removes any row where content or label is empty
df = df[['content', 'label']].dropna()

print("\nAfter combining title+text, shape:", df.shape)

# ============================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================

# Create a stemmer object — we will use this inside our function
stemmer = PorterStemmer()

# Load English stopwords into a Python set (faster lookup than list)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Clean a single news article.
    Input : raw text string  e.g. "Breaking NEWS: Govt ANNOUNCES policy!!"
    Output: cleaned string   e.g. "break news govern announc polici"
    """

    # 1. Lowercase everything
    #    "Breaking NEWS" → "breaking news"
    text = text.lower()

    # 2. Remove everything that is NOT a letter or space
    #    re.sub(pattern, replacement, string)
    #    [^a-z ] means "any character that is not a-z or space"
    #    We replace those characters with '' (nothing = delete them)
    #    "breaking news: govt!!" → "breaking news govt"
    text = re.sub(r'[^a-z ]', '', text)

    # 3. Split into individual words
    #    "breaking news govt" → ["breaking", "news", "govt"]
    words = text.split()

    # 4. Remove stopwords AND stem each word — done in one line
    #    For each word: keep it only if NOT a stopword, then stem it
    #    stemmer.stem("breaking") → "break"
    #    stemmer.stem("government") → "govern"
    words = [stemmer.stem(w) for w in words if w not in stop_words]

    # 5. Join words back into a single string
    #    ["break", "news", "govern"] → "break news govern"
    return ' '.join(words)

# Apply our preprocess function to every row in the 'content' column
# .apply() runs the function once for each article — all 6335 of them
print("\nPreprocessing text... (this takes ~30 seconds)")
df['content'] = df['content'].apply(preprocess)
print("Preprocessing done!")

# Show a sample of what preprocessed text looks like
print("\nSample preprocessed article:")
print(df['content'].iloc[0][:200])

# ============================================================
# STEP 4: CONVERT TEXT TO NUMBERS USING TF-IDF
# ============================================================

# TF-IDF stands for Term Frequency - Inverse Document Frequency
# It converts each article into a vector (list) of numbers
# Each number represents how important a word is in that article
#
# max_features=5000 means: only use the top 5000 most important words
# (Using all words would be too slow and cause noise)

tfidf = TfidfVectorizer(max_features=5000)

# X = our features (the processed article text, as numbers)
# y = our labels (FAKE or REAL)

X = tfidf.fit_transform(df['content'])
# fit_transform does two things:
# fit    → learns which 5000 words are most important across all articles
# transform → converts every article into a row of 5000 numbers

y = df['label']
# y is a column of "FAKE" and "REAL" strings

print("\nTF-IDF matrix shape:", X.shape)
# e.g. (6335, 5000) → 6335 articles, each represented by 5000 numbers

# ============================================================
# STEP 5: SPLIT INTO TRAINING AND TESTING DATA
# ============================================================

# We split the data:
# 80% goes to TRAINING   → model learns from this
# 20% goes to TESTING    → we test model on data it has never seen
#
# test_size=0.2 means 20% for testing
# random_state=42 means the split is reproducible (same every time)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# STEP 6: TRAIN THE LOGISTIC REGRESSION MODEL
# ============================================================

# Create the model
# max_iter=1000 means: try up to 1000 times to find the best parameters
model = LogisticRegression(max_iter=1000)

# .fit() is where the actual LEARNING happens
# We show the model all training articles + their labels
# It finds patterns: which words appear more in FAKE vs REAL news
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================

# Use the trained model to predict labels for the TEST articles
# These are articles the model has NEVER seen before
y_pred = model.predict(X_test)

# Compare predictions to actual labels
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")

# Detailed report: precision, recall, f1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix: shows how many correct/wrong predictions
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# STEP 8: SAVE THE MODEL AND VECTORIZER
# ============================================================

# We save TWO things:
# 1. The trained model (it knows how to classify fake vs real)
# 2. The TF-IDF vectorizer (it knows how to convert text to numbers)
# Both are needed when Flask receives a new article to classify

# pickle.dump saves a Python object to a file
# 'wb' means write in binary mode
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nModel saved as model.pkl")
print("TF-IDF vectorizer saved as tfidf.pkl")
print("\n✅ model.py complete! Now run app.py")