import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation

from helper_nlp import parsed_data_loading, tf_idf_sklearn, drop_certain_words, display_topics, time_series_analysis, \
    second_lda


from konlpy.tag import Komoran


# Input
save = False
tokenizer = 'kiwi'

# Analysis
baking_data, equip_data = parsed_data_loading(nouns=False, tokenizer=tokenizer)
baking_vectorizer, baking_matrix, baking_data2 = tf_idf_sklearn(baking_data)
baking_corpus_sk = np.array(baking_vectorizer.get_feature_names())

# baking_drop_words = ['아시', '보신', '가요', '건가', '구우', '안나', '정말', '일반', '움색', '하나요', '그냥', '보고', '자꾸']
baking_drop_words = ['ㅋㅋ', 'ㅋㅋㅋ', 'ㅎㅎ', 'ㅜㅜ', 'ㅠㅜ', 'ㅠㅠ', 'ㅠㅠㅠ', 'ㅠㅠㅠㅠ']
baking_corpus_sk, baking_matrix = drop_certain_words(baking_corpus_sk, baking_matrix, baking_drop_words)

if save:
    lda_sk = LatentDirichletAllocation(n_components=5)
    lda_sk.fit(baking_matrix)
    pickle.dump(lda_sk, open('lda_model_sk2.p', 'wb'))

else:
    lda_sk = pickle.load(open('lda_model_sk2.p', 'rb'))

topics = display_topics(lda_sk, baking_corpus_sk, 10)
topics_df = pd.DataFrame(topics)

topic_dist = lda_sk.transform(baking_matrix)
baking_data['topic label'] = topic_dist.argmax(1)
baking_data['topic prob'] = topic_dist.max(1)

if save:
    gdrive_path = '~/hyryou94/gdrive/SharedDrives/HandaProjects/BTM_results'
    topics_df.to_csv(os.path.join(gdrive_path, 'topics2.csv'), encoding='ms949')
    for each_topic in range(len(topics_df)):
        baking_data[['topic prob', '제목', '본문', '댓글']][baking_data['topic label'] == each_topic].to_excel(
            os.path.join(gdrive_path, 'clustered/text_%d.xlsx') % each_topic, encoding='ms949')

whole_period, monthly = time_series_analysis(baking_data)
each_cluster_data, each_cluster_topic = second_lda(baking_data, cluster_num=0)
