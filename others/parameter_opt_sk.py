import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation

from helper_nlp import parsed_data_loading, tf_idf_sklearn

baking_data, equip_data = parsed_data_loading()
baking_vectorizer, baking_matrix, baking_data2 = tf_idf_sklearn(baking_data)
baking_corpus_sk = np.array(baking_vectorizer.get_feature_names())

# Define Search Param
search_params = {'n_components': np.arange(5, 40)}

# Init the model
lda = LatentDirichletAllocation(learning_decay=0.9)

# Init Grid Search class
model = GridSearchCV(lda, search_params)
model.fit(baking_matrix)
best_lda_model = model.best_estimator_
print("Best model's params: ", model.best_params_)
print("Best log likelihood score: ", model.best_score_)
print("Model perplexity: ", best_lda_model.perplexity(baking_matrix))
