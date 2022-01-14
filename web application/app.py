from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template
import scipy
import os
import pandas as pd
#import keras.backend.tensorflow_backend as tb
#from tensorflow.keras.models import model_from_json
from keras.models import model_from_json
import pickle


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)


with open('sentence_embed.pkl','rb') as f:
    sentence_embeddings = pickle.load(f)
input_df = pd.read_csv('data.csv')
input_df = input_df.head(20000)
sentences = input_df['headline_text'].values.tolist()

model = SentenceTransformer('bert-base-nli-mean-tokens')


def performSearch(query):
	queries = [query]
	query_embeddings = model.encode(queries)

	# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
	number_top_matches = 3 #@param {type: "number"}

	print("Semantic Search Results")
	results = []
	for query, query_embedding in zip(queries, query_embeddings):
		distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

		results = zip(range(len(distances)), distances)
		results = sorted(results, key=lambda x: x[1])

    	
	return results

@app.route("/semanticsearch",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		query = request.form.get('query')
		print(query)
		results = performSearch(query)
		return render_template('semantic_search.html', query=query, results=results, sentences=sentences)
	else:
		return render_template('semantic_search.html', review="" ,results=None)
	

if __name__ == "__main__":
    app.run(debug=True)