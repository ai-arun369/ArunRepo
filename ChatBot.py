import nltk
import random
import string

## nltk.download()
nltk.data.clear_cache()
nltk.download('punkt')
nltk.download('wordnet')

# Sample text as knowledge base
chat_corpus = """
Hello! I am your friendly chatbot.
I can help you with your questions about programming.
Python is a great programming language.
Chatbots are fun to build and use.
I love helping people learn coding.
"""

# Convert to lower case
chat_corpus = chat_corpus.lower()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

sent_tokens = sent_tokenize(chat_corpus)
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in string.punctuation]
    return tokens


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_response(user_input):
    user_input = user_input.lower()
    sent_tokens.append(user_input)

    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)

    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = similarity_scores.argsort()[0][-1]
    flat = similarity_scores.flatten()
    flat.sort()
    best_score = flat[-1]

    sent_tokens.pop()  # Remove user input from list

    if best_score == 0:
        return "I'm sorry, I don't understand."
    else:
        return sent_tokens[idx]

print("Bot: Hi! I'm your chatbot. Type 'bye' to exit.")

while True:
    user_input = input("Arun: ")
    if user_input.lower() == 'bye':
        print("Bot: Goodbye! 👋")
        break
    else:
        print("Bot:", get_response(user_input))

