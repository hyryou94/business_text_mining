import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gensim
from gensim.models.ldamodel import LdaModel

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, tf_idf_gensim, pass_opt, \
    lda_param_opt, doc_labeling, time_series_analysis, second_lda

# Initial_setting Settings
os.environ['JAVA_HOME'] = '/usr/bin/java'

# Parameters
parsed = True
parameter_optimization = False
save = False

# Data Loading
if not parsed:
    baking_data, equip_data = data_loading()

    # 텍스트 클렌징
    baking_data = tokenization(cleansing(baking_data))
    baking_data.to_json('parsed_data/parsed_baking.json', orient='table')

    equip_data = tokenization(cleansing(equip_data))
    equip_data.to_json('parsed_data/parsed_equip.json', orient='table')
else:
    baking_data, equip_data = parsed_data_loading()

# TF-IDF
baking_corpus, baking_dictionary, baking_data = tf_idf_gensim(baking_data, drop_one_letter=False)

# LDA 파라미터 최적화
if parameter_optimization:
    pass_result = pass_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                           min_pass=1, max_pass=40)
    pass_result.to_csv('parameter_pass.csv', encoding='ms949')
    topic_num_result = lda_param_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                                     min_topic=2, max_topic=40)
    topic_num_result.to_csv('parameter_topic.csv', encoding='ms949')

# LDA model
if save:
    lda = gensim.models.LdaMulticore(baking_corpus, num_topics=7, id2word=baking_dictionary, passes=7, workers=16)
else:
    lda = gensim.models.LdaMulticore.load('saved_model.lda')

baking_data, topics = doc_labeling(baking_data, baking_corpus, lda)

# Time Series
whole_period, monthly = time_series_analysis(baking_data)

# Save Models and Topics
if save:
    lda.save('saved_model.lda')
    topics.to_csv('topics.csv', encoding='ms949')
    for each_topic in range(0, len(topics)):
        baking_data[['topic prob', '제목', '본문', '댓글']][baking_data['topic label'] == each_topic].to_excel(
            'clustered_text/%d.xlsx' % each_topic, encoding='ms949')


# Second LDA
each_cluster_data = second_lda(baking_data, cluster_num=0)


# keep_data[['second topic label', 'second topic prob', '제목', '본문', '댓글']].to_excel('보관.xlsx', encoding='ms949')
