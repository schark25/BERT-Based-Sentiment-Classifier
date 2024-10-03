import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import os
from sklearn.neural_network import MLPClassifier
import sys
import torch
from transformers import BertTokenizer, BertModel
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_bert_embedding(sentence_list, pooling_strategy='cls'):
    embedding_list = []
    for nn, sentence in enumerate(sentence_list):
        # if (nn%100==0)&(nn>0):
        #     print('Done with %d sentences'%nn)
        
        # Tokenize the sentence and get the output from BERT
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Take the embeddings from the last hidden state (optionally, one can use pooling techniques for different representations)
        # Here, we take the [CLS] token representation as the sentence embedding
        last_hidden_states = outputs.last_hidden_state[0]
        
        # Pooling strategies
        if pooling_strategy == "cls":
            sentence_embedding = last_hidden_states[0]
        elif pooling_strategy == "mean":
            sentence_embedding = torch.mean(last_hidden_states, dim=0)
        elif pooling_strategy == "max":
            sentence_embedding, _ = torch.max(last_hidden_states, dim=0)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        embedding_list.append(sentence_embedding)
    return torch.stack(embedding_list)


data_dir = 'data_reviews'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

N, n_cols = x_train_df.shape
y_train_df = y_train_df.values.ravel()

tr_text_list = x_train_df['text'].values.tolist()
te_text_list = x_test_df['text'].values.tolist()

tr_embedding = get_bert_embedding(tr_text_list)

te_embedding = get_bert_embedding(te_text_list)

my_bert_classifier_pipeline = Pipeline([
    ('my_classifier', MLPClassifier(max_iter = 1000, alpha = .0001, hidden_layer_sizes=(100,))),
])

my_parameter_grid_by_name = dict()
my_parameter_grid_by_name['my_classifier__alpha'] = np.logspace(-3, 3, 30) 


my_scoring_metric_name = 'roc_auc'

K = 10
skf = StratifiedKFold(n_splits=K, random_state=1234, shuffle = True)

grid_searcher = GridSearchCV(
    my_bert_classifier_pipeline,
    my_parameter_grid_by_name,
    scoring=my_scoring_metric_name,
    cv=skf,
    refit=True,
    return_train_score=True)
grid_searcher.fit(tr_embedding, y_train_df)

gsearch_results_df = pd.DataFrame(grid_searcher.cv_results_).copy()

param_keys = ['param_my_classifier__alpha']

gsearch_results_df.sort_values(param_keys, inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
# print(grid_searcher.best_params_)
print(gsearch_results_df)
bestModel = grid_searcher.best_estimator_
bestmodelresults = bestModel.predict_proba(te_embedding)[:, 1]
sys.stdout = open("yproba2_test.txt", "w")
for result in bestmodelresults:
    print(result)
