# 📰 Fake News Detection Using NLP

A machine learning-powered web application that detects whether a news article is **Real** or **Fake** using Natural Language Processing (NLP) techniques. The application provides an intuitive web interface built with Flask, enabling users to classify news content instantly.

---

## 📌 Project Overview

Fake news has become a major challenge in today's digital world. This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **Real** or **Fake** based on their textual content.

The model preprocesses the input text, converts it into numerical features using **TF-IDF Vectorization**, and predicts the result using a trained machine learning model.

---

## ✨ Features

- 🔍 Detects Fake and Real News
- 🧠 Machine Learning-based Prediction
- 📝 Text Preprocessing using NLP
- 📊 TF-IDF Feature Extraction
- ⚡ Fast and Accurate Predictions
- 🌐 User-friendly Flask Web Interface
- 📱 Responsive Design
- 🚀 Lightweight and Easy to Deploy

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn

### NLP
- NLTK
- TF-IDF Vectorizer

### Web Framework
- Flask

### Frontend
- HTML5
- CSS3

### Data Processing
- Pandas
- NumPy

---

## 📂 Project Structure

```
FakeNewsDetector/
│
├── app.py
├── model.py
├── model.pkl
├── tfidf.pkl
├── news.csv
├── templates/
│   └── index.html
├── static/
├── screenshots/
├── presentation/
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. User enters a news article.
2. Text is cleaned and preprocessed.
3. TF-IDF converts text into numerical vectors.
4. The trained Machine Learning model predicts whether the article is Real or Fake.
5. The result is displayed instantly on the web page.

---

## 🧠 Machine Learning Pipeline

```
Input News
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Trained ML Model
      │
      ▼
Prediction
      │
      ▼
Real / Fake
```

---

## 📊 Dataset

The project uses a labeled Fake News dataset containing news articles categorized as:

- Real News
- Fake News

The dataset is preprocessed before training the model.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/akarshyadav18/Fake-news-detection.git
```

Go to the project folder

```bash
cd Fake-news-detection
```

Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 📦 Requirements

- Python 3.9+
- Flask
- Scikit-learn
- Pandas
- NumPy
- NLTK

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Future Improvements

- Deep Learning (LSTM/BERT)
- Real-time News API Integration
- User Authentication
- News Source Credibility Score
- Explainable AI Predictions
- Mobile Application
- Multi-language Fake News Detection

---

## 📷 Screenshots

Add screenshots inside the `screenshots/` folder and display them here.

Example:

```markdown
![Home Page](screenshots/home.png)

![Prediction](screenshots/prediction.png)
```

---

## 🎯 Learning Outcomes

- Natural Language Processing
- Text Classification
- Machine Learning
- Flask Web Development
- Model Deployment
- Feature Engineering
- TF-IDF Vectorization
- Git & GitHub

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Create a Pull Request

---

## 👨‍💻 Author

### Akarsh Yadav

BCA Student | AI Automation Enthusiast | Full Stack Developer

**LinkedIn**
> https://www.linkedin.com/in/akarsh-yadav18/

**GitHub**
> https://github.com/akarshyadav18

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. It helps others discover the project and motivates further improvements.

---
