from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

with open("intents.json") as file:
    intents = json.load(file)

patterns = []
tags = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = np.array(tags)

model = MultinomialNB()
model.fit(X, y)

def predict_intent(user_input):
    input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(input_vectorized)
    return prediction[0]

def get_response(intent):
    for intent_data in intents["intents"]:
        if intent_data["tag"] == intent:
            return random.choice(intent_data["responses"])

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    intent = predict_intent(user_input)
    response = get_response(intent)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
