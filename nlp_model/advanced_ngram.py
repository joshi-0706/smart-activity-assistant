from collections import defaultdict, Counter
import nltk

# Download dataset (runs only first time)
nltk.download('brown')

from nltk.corpus import brown


class NgramModel:
    def __init__(self):
        self.bigram = defaultdict(list)
        self.trigram = defaultdict(list)
        self.vocab = set()

    def train(self, text):
        words = text.lower().split()
        self.vocab = set(words)

        # Build bigrams
        for i in range(len(words) - 1):
            self.bigram[words[i]].append(words[i + 1])

        # Build trigrams
        for i in range(len(words) - 2):
            key = (words[i], words[i + 1])
            self.trigram[key].append(words[i + 2])

    # Bigram with Laplace smoothing
    def predict_bigram(self, word):
        results = []
        total_vocab = len(self.vocab)

        if word in self.bigram:
            counts = Counter(self.bigram[word])
            total_count = sum(counts.values())

            for w in self.vocab:
                count = counts.get(w, 0)
                prob = (count + 1) / (total_count + total_vocab)
                results.append((w, prob))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:3]

        return []

    # Trigram with Laplace smoothing
    def predict_trigram(self, word1, word2):
        results = []
        total_vocab = len(self.vocab)

        key = (word1, word2)

        if key in self.trigram:
            counts = Counter(self.trigram[key])
            total_count = sum(counts.values())

            for w in self.vocab:
                count = counts.get(w, 0)
                prob = (count + 1) / (total_count + total_vocab)
                results.append((w, prob))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:3]

        return []


# ---------------- CUSTOM SENTENCES ----------------
custom_text = """
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
the boy is running very fast
the girl is sitting near the window
workers are walking towards the office
players are running on the field
everyone is sitting silently in the hall
the teacher is walking into the classroom
"""

# ---------------- REAL DATASET ----------------
brown_words = brown.words()[:5000]
brown_text = " ".join(brown_words)

# Combine both
text_data = custom_text + " " + brown_text

# Train model
model = NgramModel()
model.train(text_data)

print("🔹 Advanced NLP Model (Bigram + Trigram + Laplace Smoothing)")
print("Type 'exit' to stop\n")

# Interactive prediction
while True:
    text = input("Enter text: ").lower()

    if text == "exit":
        break

    words = text.split()

    if len(words) >= 2:
        result = model.predict_trigram(words[-2], words[-1])
        print("Trigram Suggestions:", result)
    elif len(words) == 1:
        result = model.predict_bigram(words[-1])
        print("Bigram Suggestions:", result)
    else:
        print("Enter valid input")