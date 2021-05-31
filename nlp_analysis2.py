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
baking_vectorizer, baking_matrix, baking_data = tf_idf_sklearn(baking_data)
baking_corpus_sk = np.array(baking_vectorizer.get_feature_names())

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
# each_cluster_data, each_cluster_topic = second_lda(baking_data, cluster_num=4)

# RQ2
baking_data['조회수'] = baking_data['조회수'].apply(
    lambda x: int(x.replace('조회 ', '').replace(',', '').replace('.', '').replace('만', '000')))
sorted_baking_data = baking_data.sort_values('조회수', ascending=False).copy()
top2000 = sorted_baking_data[:2000]
top2000_vectorizer, top2000_matrix, top2000_data = tf_idf_sklearn(top2000)
top2000_corpus_sk = np.array(top2000_vectorizer.get_feature_names())

if save:
    lda_sk2000 = LatentDirichletAllocation(n_components=7)
    lda_sk2000.fit(top2000_matrix)
    pickle.dump(lda_sk2000, open('lda_model_sk_top2000.p', 'wb'))

else:
    lda_sk2000 = pickle.load(open('lda_model_sk_top2000.p', 'rb'))

top2000_topics = display_topics(lda_sk2000, top2000_corpus_sk, 10)
top2000_topics_df = pd.DataFrame(top2000_topics)

top2000_topic_dist = lda_sk2000.transform(top2000_matrix)
top2000_data['topic label'] = top2000_topic_dist.argmax(1)
top2000_data['topic prob'] = top2000_topic_dist.max(1)

top2000_data[top2000_data['topic label'] == 5].sort_values('topic prob', ascending=False)['제목'][:40]# [40:80]
# 1 : 창업, 수업, 가게
# 2 : 버터, 크림치즈, 휘핑크림 선택
# 3 : 재료, 완성품 보관방법
# 4 : 아몬드가루, 슈가파우더, 마카롱
# 5 : 실패원인

top2000_data['count'] = np.ones(len(top2000_data))
views = top2000_data[['topic label', '조회수']].groupby('topic label').sum()
num_docs = top2000_data[['topic label', 'count']].groupby('topic label').sum()

views/num_docs.values
