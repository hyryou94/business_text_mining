import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from helper_nlp import data_loading, cleansing, tokenization, tf_idf_sklearn, count_sklearn, analysis, drop_certain_words, \
    display_topics, time_series_analysis, doc_labeling, closer_look

# Input
tokenizer = 'kiwi'

# Analysis
baking_data = data_loading() # data loading
baking_data = tokenization(cleansing(baking_data)) # cleansing and tokenizing

# Frequency analysis
baking_count, baking_dtm = count_sklearn(baking_data)
count_corpus_sk = baking_count.get_feature_names()
frequency = pd.DataFrame(baking_dtm.sum(0), columns=count_corpus_sk).T.sort_values(0, ascending=False)

# Method 1
baking_data, topics_df, lda_sk = analysis(baking_data, n_topics=20)

# Method 2
top_filtering = [True if each_label in [14, 16, 0, 5, 2] else False for each_label in baking_data['topic label']]
top_topics = baking_data[top_filtering]

whole_period, monthly_percentage, time_series_result = time_series_analysis(top_topics)
plt.figure(figsize=(10, 5))
plt.plot(time_series_result.loc['2018-06-01':'2021-04-30'].resample('M').sum().sum(1))
plt.show()

print(time_series_result.sum(1).describe())

# Method 3
sorted_baking_data = baking_data.sort_values('조회수', ascending=False).copy()
top2000_data = sorted_baking_data[:2000]
top2000_data, top2000_topics_df, lda_sk2000 = analysis(top2000_data, n_topics=5)

# Additional
top2000_data['count'] = np.ones(len(top2000_data))
views = top2000_data[['topic label', '조회수']].groupby('topic label').sum()
views.index = ['창업, 수업, 가게', '버터, 크림치즈, 휘핑크림 선택', '재료, 완성품 보관방법', '아몬드가루, 슈가파우더, 마카롱', '실패원인']
num_docs = top2000_data[['topic label', 'count']].groupby('topic label').sum()

# Qualitative Analysis
# filters는 클래스 데이터에서 창업관련을 제외하고 레시피 추천 데이터를 추출하기 위하여 사용
# word_filter1 = [True if ('창업' not in content) and ('가게' not in content) else False for content in each_topic_df['본문']]
# word_filter2 = [True if ('레시피' in content) and ('추천' in content) else False for content in each_topic_df['본문']]
# word_filter1_2000 = [True if ('창업' not in content) and ('가게' not in content) else False for content in
#                      each_topic2000_df['본문']]
# word_filter2_2000 = [True if ('창업' not in content) and ('가게' not in content) else False for content in
#                      each_topic2000_df['제목']]

closer_look(baking_data, topic_num=2, content='제목')
closer_look(top2000_data, topic_num=2, content='제목')

