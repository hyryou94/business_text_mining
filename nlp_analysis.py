import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gensim
from gensim.models.ldamodel import LdaModel

from sklearn.feature_extraction.text import TfidfVectorizer

from helper_nlp import data_loading, parsed_data_loading, cleansing, tokenization, tf_idf, pass_opt, lda_param_opt

# Initial_setting Settings
os.environ['JAVA_HOME'] = '/usr/bin/java'

# Parameters
parsed = True
parameter_optimization = False

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
baking_corpus, baking_dictionary, baking_data = tf_idf(baking_data, drop_one_letter=False)
#equip_corpus, equip_dictionary, equip_data = tf_idf(equip_data, drop_one_letter=True)

# LDA 파라미터 최적화
if parameter_optimization:
    pass_result = pass_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                           min_pass=1, max_pass=40)
    pass_result.to_csv('parameter_pass.csv', encoding='ms949')
    topic_num_result = lda_param_opt(corpus=baking_corpus, dictionary=baking_dictionary, data=baking_data,
                                     min_topic=2, max_topic=40)
    topic_num_result.to_csv('parameter_topic.csv', encoding='ms949')

# LDA
lda = gensim.models.LdaMulticore(baking_corpus, num_topics=20, id2word=baking_dictionary, passes=20, workers=16)

# Topic 나누기
lda_topics = sorted(lda.print_topics(num_words=10), key=lambda x: x[0])
topics = pd.DataFrame(
    [[word.split('*')[1].replace('\"', '') for word in topic[1].split(' + ')] for topic in lda_topics])

baking_data['topic'] = [lda[each_doc] for each_doc in baking_corpus]
baking_data['label w/ prob'] = baking_data['topic'].apply(lambda x: max(x, key=lambda x: x[1]))
baking_data['topic label'] = baking_data['label w/ prob'].apply(lambda x: x[0])
baking_data['topic prob'] = baking_data['label w/ prob'].apply(lambda x: x[1])

topics.to_csv('topics.csv', encoding='ms949')

# 날짜 변환
baking_data['날짜'] = pd.to_datetime(baking_data['날짜'])
period_baking_data = baking_data.copy()
period_baking_data.index = period_baking_data['날짜'].values
period_baking_data['날짜'] = [date.date() for date in period_baking_data.index]
period_baking_data['count'] = np.ones(len(period_baking_data))

whole_period = period_baking_data[['topic label', 'count']].groupby('topic label').sum().sort_values('count',
                                                                                                     ascending=False)
whole_period_percent = whole_period/whole_period.sum()

time_series = period_baking_data[['날짜', 'topic label', 'count']].groupby(['날짜', 'topic label']).sum().reset_index()
time_series_result = time_series.pivot_table(columns=['topic label'], index=['날짜'], values='count').fillna(0)
time_series_result.index = pd.to_datetime(time_series_result.index)
monthly_result = time_series_result.resample('3M').sum()
monthly_percentage = (monthly_result.T / monthly_result.sum(1)).T

plt.plot(monthly_percentage)
plt.legend(monthly_percentage.columns)
plt.show()

plt.plot(monthly_percentage[18])
plt.legend([monthly_percentage[18].name])
plt.show()

#
for each_topic in range(0, 19):
    baking_data[['topic prob', '제목', '본문', '댓글']][baking_data['topic label'] == each_topic].to_excel(
        'clustered_text/%d.xlsx' % each_topic, encoding='ms949')

keep_data = baking_data[baking_data['topic label'] == 18].copy()
keep_data['text_tokenized'] = keep_data['title_tokenized'] + keep_data['text_tokenized']
keep_corpus, keep_dictionary, keep_data = tf_idf(keep_data, drop_one_letter=False)
second_lda = gensim.models.LdaMulticore(keep_corpus, num_topics=10, id2word=keep_dictionary, passes=20, workers=16)

second_lda_topics = sorted(second_lda.print_topics(num_words=10), key=lambda x: x[0])
second_topics = pd.DataFrame(
    [[word.split('*')[1].replace('\"', '') for word in topic[1].split(' + ')] for topic in second_lda_topics])

keep_data['second topic'] = [second_lda[each_doc] for each_doc in keep_corpus]
keep_data['second label w/ prob'] = keep_data['second topic'].apply(lambda x: max(x, key=lambda x: x[1]))
keep_data['second topic label'] = keep_data['second label w/ prob'].apply(lambda x: x[0])
keep_data['second topic prob'] = keep_data['second label w/ prob'].apply(lambda x: x[1])

keep_data[['second topic label', 'second topic prob', '제목', '본문', '댓글']].to_excel('보관.xlsx', encoding='ms949')
