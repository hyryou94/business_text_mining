import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation

from helper_nlp import parsed_data_loading, tf_idf_sklearn, count_sklearn, analysis, drop_certain_words, \
    display_topics, time_series_analysis

# Input
save = False
tokenizer = 'kiwi'

# Analysis
baking_data, equip_data = parsed_data_loading(nouns=False, tokenizer=tokenizer)

# Method 1
baking_count, baking_dtm = count_sklearn(baking_data)
count_corpus_sk = baking_count.get_feature_names()
frequency = pd.DataFrame(baking_dtm.sum(0), columns=count_corpus_sk).T.sort_values(0, ascending=False)
baking_data, topics_df, lda_sk = analysis(baking_data, save)

top_filtering = [True if each_label in [14, 16, 0, 5, 2] else False for each_label in baking_data['topic label']]
top_topics = baking_data[top_filtering]

# Method 2
whole_period, monthly_percentage, time_series_result = time_series_analysis(top_topics)

plt.figure(figsize=(10, 5))
plt.plot(time_series_result.loc['2018-06-01':'2021-04-30'].resample('M').sum().sum(1))
plt.show()

time_series_result.sum(1).describe()

# Method 3
top_2000_analysis
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

# 0 : 창업, 수업, 가게
# 1 : 버터, 크림치즈, 휘핑크림 선택
# 2 : 재료, 완성품 보관방법
# 3 : 아몬드가루, 슈가파우더, 마카롱
# 4 : 실패원인

top2000_data['count'] = np.ones(len(top2000_data))
views = top2000_data[['topic label', '조회수']].groupby('topic label').sum()
views.index = ['창업, 수업, 가게', '버터, 크림치즈, 휘핑크림 선택', '재료, 완성품 보관방법', '아몬드가루, 슈가파우더, 마카롱', '실패원인']
num_docs = top2000_data[['topic label', 'count']].groupby('topic label').sum()

# views/num_docs.values

# Result save
gdrive_path = '~/hyryou94/gdrive/SharedDrives/HandaProjects/BTM_results'
if save:
    topics_df.to_csv(os.path.join(gdrive_path, 'topics20.csv'), encoding='ms949')
    top2000_topics_df.to_csv(os.path.join(gdrive_path, 'topics_top2000.csv'), encoding='ms949')
    frequency.to_csv('frequency.csv', encoding='ms949')

    for each_topic in range(len(topics_df)):
        baking_data[['topic prob', '제목', '본문', '댓글']][baking_data['topic label'] == each_topic].to_excel(
            os.path.join(gdrive_path, 'clustered/text_%d.xlsx') % each_topic, encoding='ms949')

# Qualitative Aggregation
# Method 1
# [14, 16, 0, 5, 2]
topic_num = 5
each_topic_df = baking_data[baking_data['topic label'] == topic_num]
word_filter1 = [True if ('창업' not in content) and ('가게' not in content) else False for content in each_topic_df['본문']]
word_filter2 = [True if ('레시피' in content) and ('추천' in content) else False for content in each_topic_df['본문']]
print(each_topic_df[['topic prob', '제목', '본문', '댓글']].sort_values(by='topic prob', ascending=False)['제목'][:40])

#each_topic_df[['topic prob', '제목', '본문', '댓글']].sort_values(by='topic prob', ascending=False).to_excel(
#    os.path.join(gdrive_path, 'mod topic full %d.xlsx' % topic_num), encoding='ms949')

# Method 2
each_topic2000_df = top2000_data[top2000_data['topic label'] == 0].sort_values('topic prob', ascending=False)
word_filter1_2000 = [True if ('창업' not in content) and ('가게' not in content) else False for content in
                     each_topic2000_df['본문']]
word_filter2_2000 = [True if ('창업' not in content) and ('가게' not in content) else False for content in
                     each_topic2000_df['제목']]
each_topic2000_df[['topic prob', '제목', '본문', '댓글']].sort_values(by='topic prob', ascending=False)[
    word_filter1_2000].to_excel(os.path.join(gdrive_path, 'mod topic full2000 0.xlsx'), encoding='ms949')
