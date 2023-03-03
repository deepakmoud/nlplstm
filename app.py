import numpy as np
from flask import Flask, request, jsonify, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Initialize the stemmer
ps = PorterStemmer()

# Sample news

voc_size = 5000


nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
# Download the WordNet corpus
nltk.download('wordnet')
nltk.download('omw-1.4')
# Create a lemmatizer object
lemmatizer = nltk.WordNetLemmatizer()
# Loading saved model from Drive.
from tensorflow.keras.models import load_model
model = load_model('nlplstm.h5')
app = Flask(__name__)



@app.route('/')
def welcome():
  
    return render_template("index.html")


@app.route('/review',methods=['GET'])
def review():
  text = request.args.get('text')
    # Preprocess the news
  review = re.sub('[^a-zA-Z]', ' ', text)
  review = review.lower()
  review = review.split()
    
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  onehot_repr = [one_hot(words, voc_size) for words in review.split()]

  # Pad the sequences
  sent_length = 100
  embedded_docs = pad_sequences(onehot_repr, padding='post', maxlen=sent_length)

  # Predict the reliability of the news
  y_pred = model.predict(embedded_docs)

  y_pred = np.where(y_pred > 0.5, 1, 0)

  # Print the result
  if np.all(y_pred == 1):
    result = "Review is Positive"
    print("Review is positive")
  else:
    result = "Review is negative"
    print("Review is Negative")
  return render_template('index.html', prediction_text='Movie Review Anlaysis: {}'.format(result))

if __name__=="__main__":
  app.run()
