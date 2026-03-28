from collections import defaultdict

class BigramModel:
    def __init__(self):
        self.bigrams = defaultdict(list)

    def train(self, text):
        words = text.lower().split()
        for i in range(len(words) - 1):
            self.bigrams[words[i]].append(words[i + 1])

    def predict(self, word):
        if word in self.bigrams:
            return self.bigrams[word][0]
        return "..."

text_data = """
i am walking in the park
i am sitting on the chair
he is running fast
she is walking slowly
they are sitting together
"""

model = BigramModel()
model.train(text_data)

while True:
    word = input("Enter a word: ")
    print("Next word:", model.predict(word))