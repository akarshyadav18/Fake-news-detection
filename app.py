# ============================================================
# FILE: app.py
# PURPOSE: Flask web server — receives news article from user,
#          runs it through our model, returns FAKE or REAL
# ============================================================

# flask: web framework — lets us create a website with Python
# Flask: the main class that creates our web application
# render_template: loads HTML files from the templates/ folder
# request: lets us read data submitted by the user (the news text)
from flask import Flask, render_template, request

# pickle: loads our saved model and vectorizer from .pkl files
import pickle

# re: for cleaning the user's input text (same as in model.py)
import re

# nltk tools — same ones we used in model.py
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download nltk data (in case it's not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ============================================================
# STEP 1: CREATE THE FLASK APP
# ============================================================

# Flask(__name__) creates our web application
# __name__ tells Flask where to find files (templates, etc.)
app = Flask(__name__)

# ============================================================
# STEP 2: LOAD THE SAVED MODEL AND VECTORIZER
# ============================================================

# 'rb' means read in binary mode
# We load BOTH the model and the vectorizer
# The vectorizer must be the SAME one used during training
# because it knows which 5000 words to look for

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

print("Model and vectorizer loaded successfully!")

# ============================================================
# STEP 3: SET UP TEXT PREPROCESSING (same as model.py)
# ============================================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Clean user-submitted text exactly the same way
    we cleaned the training data. This is critical —
    if we clean differently, the model gets confused.
    """
    # Lowercase
    text = text.lower()
    # Remove non-letters
    text = re.sub(r'[^a-z ]', '', text)
    # Split into words
    words = text.split()
    # Remove stopwords and stem
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    # Rejoin into string
    return ' '.join(words)

# ============================================================
# STEP 4: DEFINE ROUTES (PAGES OF OUR WEBSITE)
# ============================================================

# @app.route('/') means: when user visits the homepage
# methods=['GET', 'POST'] means this page handles both:
#   GET  → user just visits the page (show empty form)
#   POST → user submitted the form (make prediction)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main page of our web app.
    Shows a text box where user pastes a news article.
    After submission, shows prediction result.
    """

    # These variables will be sent to the HTML template
    result = None      # Will be "FAKE" or "REAL"
    news_text = ""     # Will hold what the user typed
    confidence = None  # Will hold the confidence percentage

    # request.method tells us HOW the page was accessed
    if request.method == 'POST':

        # request.form['news'] gets the text from the input box
        # 'news' matches the name="" attribute in our HTML form
        news_text = request.form['news']

        # Only predict if user actually typed something
        if news_text.strip():

            # Step 1: Clean the text
            cleaned = preprocess(news_text)

            # Step 2: Convert to TF-IDF numbers
            # transform() converts text to numbers using the
            # SAME vocabulary learned during training
            vectorized = tfidf.transform([cleaned])
            # We wrap cleaned in [] to make it a list of 1 article

            # Step 3: Predict using our trained model
            prediction = model.predict(vectorized)[0]
            # [0] gets the first (and only) result from the list
            # prediction is either "FAKE" or "REAL"

            # Step 4: Get confidence score (probability)
            # predict_proba returns probability for each class
            # e.g. [0.08, 0.92] means 8% FAKE, 92% REAL
            proba = model.predict_proba(vectorized)[0]

            # max(proba) gets the highest probability
            # * 100 converts 0.92 → 92
            # round(..., 2) keeps 2 decimal places
            confidence = round(max(proba) * 100, 2)

            result = prediction
            # result is now "FAKE" or "REAL"

    # render_template loads index.html from templates/ folder
    # We pass result, news_text, confidence as variables
    # The HTML file can use these variables with {{ result }} etc.
    return render_template(
        'index.html',
        result=result,
        news_text=news_text,
        confidence=confidence
    )

# ============================================================
# STEP 5: RUN THE APP
# ============================================================

# This block only runs when you directly run: python3 app.py
# debug=True means: if code changes, server restarts automatically
#                   also shows detailed error messages
# port=5000 means the website runs at http://localhost:5000

if __name__ == '__main__':
    app.run(debug=True, port=5000)