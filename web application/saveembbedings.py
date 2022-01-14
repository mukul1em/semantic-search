from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template
import scipy
import pandas as pd
import os
import pandas as pd
#import keras.backend.tensorflow_backend as tb
#from tensorflow.keras.models import model_from_json
from keras.models import model_from_json
import pickle



#tb._SYMBOLIC_SCOPE.value = True

# Load the BERT model. Various models trained on Natural Language Inference (NLI) https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md and 
# Semantic Textual Similarity are available https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md

model = SentenceTransformer('bert-base-nli-mean-tokens')

# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
#app = Flask(__name__)




@app.route("/")
def hello():
	return TEMPLATE_DIR

NEWS_FILE_NAME = r"/Users/mukulrawat/Documents/ML Projects/Semantic serach/web application/data.csv"



# def read_csv(filepath):
#      if os.path.splitext(filepath)[1] != '.csv':
#           return  # or whatever
#      seps = [',', ';', '\t']                    # ',' is default
#      encodings = [None, 'utf-8', 'ISO-8859-1']  # None is default
#      for sep in seps:
#          for encoding in encodings:
#               try:
#                   return pd.read_csv(filepath, encoding=encoding, sep=sep)
#               except Exception:  # should really be more specific 
#                   pass
#      raise ValueError("{!r} is has no encoding in {} or seperator in {}"
#                       .format(filepath, encodings, seps))


input_df = pd.read_csv(NEWS_FILE_NAME)
input_df = input_df.head(20000)
sentences = input_df['headline_text'].values.tolist()

#sentences = ['aba decides against community broadcasting licence', 
#             'act fire witnesses must be aware of defamation',
#             'a g calls for infrastructure protection summit',
#             'air nz staff in aust strike for pay rise',
#             'air nz strike to affect australian travellers',
#             'ambitious olsson wins triple jump',
#             'antic delighted with record breaking barca',
#             'aussie qualifier stosur wastes four memphis match',
#             'aust addresses un security council over iraq',
#             'australia is locked into war timetable opp',
#             'australia to contribute 10 million in aid to iraq']

# Each sentence is encoded as a 1-D vector with 78 columns

sentence_embeddings = model.encode(sentences)

with open('sentence_embed.pkl', 'wb') as f:
    pickle.dump(sentence_embeddings, f)
