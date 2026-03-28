from collections import defaultdict, Counter

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

    # Laplace smoothing for bigram
    def predict_bigram(self, word):
        results = []
        total_words = len(self.vocab)

        if word in self.bigram:
            counts = Counter(self.bigram[word])
            total_count = sum(counts.values())

            for w in self.vocab:
                count = counts.get(w, 0)
                prob = (count + 1) / (total_count + total_words)
                results.append((w, prob))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:3]

        return []

    # Laplace smoothing for trigram
    def predict_trigram(self, word1, word2):
        results = []
        total_words = len(self.vocab)

        key = (word1, word2)

        if key in self.trigram:
            counts = Counter(self.trigram[key])
            total_count = sum(counts.values())

            for w in self.vocab:
                count = counts.get(w, 0)
                prob = (count + 1) / (total_count + total_words)
                results.append((w, prob))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:3]

        return []

# Better dataset (more natural)
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
"""

# Train model
model = NgramModel()
model.train(text_data)

print("🔹 Advanced NLP Model (Bigram + Trigram + Smoothing)")
print("Type 'exit' to stop\n")

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