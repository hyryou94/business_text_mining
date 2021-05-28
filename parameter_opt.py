import os

import gensim
from gensim.models.ldamodel import LdaModel

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, tf_idf, pass_opt, lda_param_opt

baking_data, equip_data = parsed_data_loading()
baking_corpus, baking_dictionary, baking_data = tf_idf(baking_data, drop_one_letter=True)

pass_result = pass_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                       min_pass=1, max_pass=40)
pass_result.to_csv('parameter_pass.csv', encoding='ms949')
topic_num_result = lda_param_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                                 min_topic=2, max_topic=40)
topic_num_result.to_csv('parameter_topic.csv', encoding='ms949')
