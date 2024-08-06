from flask import Flask, request, jsonify, render_template
import pickle
import nltk

nltk.download('punkt')

app = Flask(__name__)

# Load your model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form.get('text', '')
        preprocessed_text = preprocess_text(text)
        X_new = vectorizer.transform([preprocessed_text])
        predicted_sentiment = model.predict(X_new)[0]
        prediction = {'text': text, 'predicted_sentiment': predicted_sentiment}
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
