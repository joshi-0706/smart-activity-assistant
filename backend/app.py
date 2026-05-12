import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict, Counter

# ---------------- INIT APP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- NLP MODEL ----------------
class NgramModel:
    def __init__(self):
        self.bigram = defaultdict(list)

    def train(self, text):
        words = text.lower().split()
        for i in range(len(words) - 1):
            self.bigram[words[i]].append(words[i + 1])

    def predict(self, word):
        if word in self.bigram:
            counts = Counter(self.bigram[word])
            return [w for w, _ in counts.most_common(3)]
        return ["no", "suggestion", "found"]

# -------- TRAIN NLP MODEL --------
text_data = """
i am walking in the park
i am walking slowly in the evening
i am sitting on the chair
i am sitting and reading a book
he is running fast in the ground
he is running daily for fitness
she is walking slowly and calmly
they are sitting together and talking
we are walking towards the park
we are running for exercise daily
people are walking in the morning
students are sitting in the classroom
the man is walking near the road
the woman is sitting quietly on the bench
children are running in the playground
he is walking with his friends
she is sitting and studying for exams
they are running in the marathon
we are sitting and watching television
people are walking in the garden area
students are studying in class
people are working in office
"""

nlp_model = NgramModel()
nlp_model.train(text_data)

# ---------------- ACTIVITY API (DISABLED IN CLOUD) ----------------
@app.route('/predict_activity', methods=['POST'])
def predict_activity():
    return jsonify({
        "activity": "Activity model not available in cloud deployment"
    })

# ---------------- NLP API ----------------
@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.json['text'].lower()
    words = text.split()

    if len(words) == 0:
        return jsonify({"suggestions": []})

    last_word = words[-1]
    suggestions = nlp_model.predict(last_word)

    return jsonify({
        "suggestions": suggestions
    })

# ---------------- HOME ROUTE ----------------
@app.route('/')
def home():
    return "Backend is running 🚀"

# ---------------- RUN SERVER (RENDER FIX) ----------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)