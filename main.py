import os
import nltk
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request
import numpy as np
from TextPreprocessing import TextPreprocessing
import Model as model
nltk.download('stopwords')
app = Flask(__name__)

# Load the tokenizer and pre-trained model
tp = TextPreprocessing("all")

# import

model_file = os.path.join(os.getcwd(), 'model', 'model_with_LSTM.h5')
myLSTM = tf.keras.models.load_model(model_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('user_input')
    processed_text = tp.preprocess_text(text)
    instance = tp.preprocess_and_tokenize_test_case(tokenizer, processed_text, "post", "post", 100)
    prediction = myLSTM.predict(instance)
    label, confidence = model.predict_label(prediction)

    # Get the probability of the predicted label
    probability = confidence.item()
    print("probability ", probability)

    return render_template('result.html', label=label, confidence=confidence, probability=probability)

if __name__ == '__main__':
    # Read the training data from the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'sentiment labelled sentences', 'all data.csv')
    print(file_path, "*" * 100)
    data = pd.read_csv(file_path, names=["Text", "emotion"], header=None)
    

    from sklearn.model_selection import train_test_split
    max_words = 10000
    oov_word = "<00V>"
    padding_type = "post"
    truncating_type = "post"
    pad_len = 100
    # Get the 'text' column as a list
    # X_train = data['Text'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(data.Text.values, data.emotion.values, test_size=0.20, random_state=42, shuffle=True)
    tokenizer, vocab_size, X_train_padded, X_test_padded = tp.tokenizer_and_pad_training(X_train, X_test, max_words, oov_word, padding_type, truncating_type, pad_len)
    # Tokenize and pad the training data
    # tokenizer, _, _, _ = tp.tokenizer_and_pad_training(X_train, None, 10000, "<OOV>", "post", "post", 100)


    app.run(host = "0.0.0.0",port = 8080,debug=True)
