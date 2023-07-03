
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import os
import urllib.request
import io

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessing:
    def __init__(self, name):
        dataset_dir = os.path.join(os.getcwd(), 'dataset')

        if name.lower() == 'amazon':
            self.path = os.path.join(dataset_dir, 'sentiment labelled sentences', 'amazon_cells_labelled.txt')
        elif name.lower() == 'yelp':
            self.path = os.path.join(dataset_dir, 'sentiment labelled sentences', 'yelp.txt')
        elif name.lower() == 'imdb':
            self.path = os.path.join(dataset_dir, 'sentiment labelled sentences', 'imdb.txt')
        elif name.lower() == 'all':
            self.path = os.path.join(dataset_dir, 'sentiment labelled sentences', 'all.csv')

        self.stop_words = stopwords.words('english')
        unwanted_stopwords = {
            'no', 'nor', 'not', 'ain', 'aren', "aren't", 'couldn', 'what', 'which', 'who', 'whom',
            'why', 'how', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
            'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
            'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't"
        }

        self.stop_words = [ele for ele in self.stop_words if ele not in unwanted_stopwords]
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.embeddings_url = 'https://example.com/glove.6B.100d.txt'  # Replace with the actual URL

    def load_glove_embeddings(self):
        response = urllib.request.urlopen(self.embeddings_url)
        embeddings_file = io.TextIOWrapper(response, encoding='utf-8')

        embeddings_index = {}
        for line in embeddings_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        return embeddings_index

    def preprocess_text(self, user_text):
        # Remove punctuations and numbers
        user_text = re.sub('[^a-zA-Z]', ' ', user_text)

        # Remove single characters
        user_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', user_text)

        # Remove multiple spaces
        user_text = re.sub(r'\s+', ' ', user_text)
        user_text = user_text.lower()

        # Convert text sentence to tokens
        user_text = word_tokenize(user_text)

        # Remove unnecessary stopwords
        filtered_text = []
        for t in user_text:
            if t not in self.stop_words:
                filtered_text.append(t)

        # Word lemmatization
        processed_text1 = []
        for t in filtered_text:
            word1 = self.wordnet_lemmatizer.lemmatize(t, pos="n")
            word2 = self.wordnet_lemmatizer.lemmatize(word1, pos="v")
            word3 = self.wordnet_lemmatizer.lemmatize(word2, pos=("a"))
            processed_text1.append(word3)

        result = " ".join(processed_text1)

        return result

    def get_data(self):
        file = open(self.path, "r")
        data = file.readlines()
        corpus = []
        labels = []

        for d in data:
            d = d.split("\t")
            corpus.append(d[0])
            labels.append(d[1].replace("\n", ""))
        file.close()

        return corpus, labels

    def count_vectorize(self, X_train, X_test):
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(X_train)

        # Transform the training and validation data using count vectorizer object
        xtrain_count = count_vect.transform(X_train)
        xvalid_count = count_vect.transform(X_test)

        return xtrain_count, xvalid_count

    def word_TF_IDF_vectorize(self, X_train, X_test):
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
        tfidf_vect.fit(X_train)

        xtrain_tfidf = tfidf_vect.transform(X_train)
        xvalid_tfidf = tfidf_vect.transform(X_test)

        return xtrain_tfidf, xvalid_tfidf

    def n_gram_TF_IDF_vectorize(self, X_train, X_test):
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                           max_features=10000)
        tfidf_vect_ngram.fit(X_train)

        xtrain_tfidf_ngram = tfidf_vect_ngram.transform(X_train)
        xvalid_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

        return xtrain_tfidf_ngram, xvalid_tfidf_ngram

    def char_TF_IDF_vectorize(self, X_train, X_test):
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                                 max_features=10000)
        tfidf_vect_ngram_chars.fit(X_train)

        xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_train)
        xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_test)

        return xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars

    def tokenizer_and_pad_training(self, X_train, X_test, max_words, oov_word, padding_type, truncating_type, pad_len):
        # Generate token sequences
        tokenizer = Tokenizer(num_words=max_words, oov_token=oov_word)
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        # Pad the sequences
        vocab_size = len(tokenizer.word_index) + 1

        X_train_padded = np.array(pad_sequences(X_train, padding=padding_type, truncating=truncating_type, maxlen=pad_len))
        X_test_padded = np.array(pad_sequences(X_test, padding=padding_type, truncating=truncating_type, maxlen=pad_len))

        return tokenizer, vocab_size, X_train_padded, X_test_padded

    def preprocess_and_tokenize_test_case(self, tokenizer, test_case, padding_type, truncating_type, pad_len):
        processed_test_case = [self.preprocess_text(test_case)]
        instance = tokenizer.texts_to_sequences(processed_test_case)
        instance = pad_sequences(instance, padding=padding_type, truncating=truncating_type, maxlen=pad_len)

        return instance

    def get_embedding_matrix(self, vocab_size, tokenizer):
        embeddings_index = self.load_glove_embeddings()

        embedding_matrix = np.zeros((vocab_size, 100))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

