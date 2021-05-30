import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import LatentDirichletAllocation

from helper_nlp import parsed_data_loading, tf_idf_gensim, tf_idf_sklearn, drop_certain_words, display_topics, \
    doc_labeling, time_series_analysis, second_lda

# Input
save = False

# Analysis
baking_data, equip_data = parsed_data_loading()
baking_vectorizer, baking_matrix, baking_data2 = tf_idf_sklearn(baking_data)
baking_corpus_sk = np.array(baking_vectorizer.get_feature_names())

baking_drop_words = ['아시', '보신', '가요', '건가', '구우', '안나', '정말', '일반', '움색', '하나요', '그냥', '보고']
baking_corpus_sk, baking_matrix = drop_certain_words(baking_corpus_sk, baking_matrix, baking_drop_words)

if save:
    lda_sk = LatentDirichletAllocation(n_components=7)
    lda_sk.fit(baking_matrix)
    pickle.dump(lda_sk, open('lda_model_sk.p', 'wb'))

else:
    lda_sk = pickle.load(open('lda_model_sk.p', 'rb'))

topics = display_topics(lda_sk, baking_corpus_sk, 10)
topics_df = pd.DataFrame(topics)

topic_dist = lda_sk.transform(baking_matrix)
baking_data['topic label'] = topic_dist.argmax(1)
baking_data['topic prob'] = topic_dist.max(1)

if save:
    topics_df.to_csv('topics2.csv', encoding='ms949')
    for each_topic in range(len(topics_df)):
        baking_data[['topic prob', '제목', '본문', '댓글']][baking_data['topic label'] == each_topic].to_excel(
            'clustered_text2/%d.xlsx' % each_topic, encoding='ms949')

whole_period, monthly = time_series_analysis(baking_data)
each_cluster_data, each_cluster_topic = second_lda(baking_data, cluster_num=0)
