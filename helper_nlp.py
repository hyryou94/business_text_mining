import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gensim
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from kiwipiepy import Kiwi
from konlpy.tag import Okt

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

os.environ['JAVA_HOME'] = '/usr/bin/java'


def data_loading():
    baking_data = pd.read_json('data_baking/baking_v2.json', orient='table').dropna().sample(2000)
    return baking_data


def clean_text(text):
    # 필요없는 부분 제거
    text = re.sub('http[s]?://\S+', '', text)  # http url 제거
    text = re.sub('\S*@\S*\s?', '', text)  # 기자 emails 제거
    text = re.sub(r'\[.*?\]', '', text)  # 대괄호안에 텍스트 제거 : 뉴스이름 + 기자이름
    text = re.sub(r"\w*\d\w*", '', text)  # 숫자 포함하는 텍스트 제거
    text = re.sub('[?.,;:|\)*~`’!^\-_+<>@\#$%&-=#}※]', '', text)  # 특수문자 이모티콘 제거
    text = re.sub("\n", '', text)  # 개행문자 제거
    text = re.sub("\xa0", '', text)  # 개행문자 제거
    text = re.sub(r'Copyright .* rights reserved', '', text)  # "Copyright all rights reserved" 제거
    return text


def preprocessing(text):
    # 특수문자나 이모티콘 등 아래의 특수기호들을 제거합니다(%등은 남김).
    text = re.sub('[?.,;:|\)*~`’!^\-_+<>@\#$%&-=#}※]', '', text)
    # 위에서 특수문자를 제거한 text에서 한글, 영문만 남기고 모두 제거하도록 합니다.
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]", ' ', text)
    return text


def cleansing(df):
    df = df.drop_duplicates(subset='본문', keep='first')
    del_content1 = '=============================\n일상의 작은 행복\n베이킹은 오븐엔조이와 함께~\n\n서로의 작은 배려로 멋진 카페를 만들어보아요 ^_^b\n\n◇오븐엔조이 기본 에티켓◇\nhttps://cafe.naver.com/delonghi/book5100010\n============================='
    del_content2 = '타카페 또는 블로그의 이벤트, 공동구매, 행사등의 공유는 금지되어있습니다.\n게시물에 주요내용이 없는 외부링크 공유 게시물은 삭제될 수 있습니다.\n원활한 카페 운영을 위해 이해와 협조 부탁드립니다.'
    df['clean title'] = df['제목'].apply(clean_text)
    df['clean text'] = df['본문'].apply(lambda x: clean_text(x.replace(del_content1, '').replace(del_content2, '')))
    df['조회수'] = df['조회수'].apply(
        lambda x: int(x.replace('조회 ', '').replace(',', '').replace('.', '').replace('만', '000')))
    return df


# 형태소 분석
def tokenization(df, nouns=True, tokenizer='kiwi'):
    if tokenizer == 'okt':
        okt = Okt()

        if nouns:
            target_title = [okt.nouns(doc) for doc in df['clean title']]
            target_text = [okt.nouns(doc) for doc in df['clean text']]

        else:
            target_title = [okt.morphs(doc) for doc in df['clean title']]
            target_text = [okt.morphs(doc) for doc in df['clean text']]
    else:
        kiwi = Kiwi(num_workers=16)
        kiwi.prepare()

        temp_title = [[each_word[0] if ('NNG' in each_word[1]) or ('NNP' in each_word[1])
                       else each_word[0] + '다' if ('VV' in each_word[1]) or ('VA' in each_word[1])
                       else None for each_word in each_doc[0][0]]
                      for each_doc in kiwi.analyze(df['clean title'], top_n=1)]

        target_title = [[each_word for each_word in each_doc if each_word] for each_doc in temp_title]

        temp_text = [[each_word[0] if ('NNG' in each_word[1]) or ('NNP' in each_word[1])
                      else each_word[0] + '다' if ('VV' in each_word[1]) or ('VA' in each_word[1])
                      else None for each_word in each_doc[0][0]]
                     for each_doc in kiwi.analyze(df['clean title'], top_n=1)]

        target_text = [[each_word for each_word in each_doc if each_word] for each_doc in temp_text]

    # 불용어 제거
    with open('nlp_data/korean_stopwords.json', encoding='utf-8') as f:
        stopwords = json.load(f)
    stopwords.extend(['에서', '고', '이다', '는', '이', '가', '한', '씨', '"', '에', '요', '걸', '더', '케', '거', '분',
                      'ㅋㅋ', 'ㅋㅋㅋ', 'ㅎㅎ', 'ㅜㅜ', 'ㅠㅜ', 'ㅠㅠ', 'ㅠㅠㅠ', 'ㅠㅠㅠㅠ'])

    df['title_tokenized'] = target_title
    df['title_tokenized'] = df['title_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])

    df['text_tokenized'] = target_text
    df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])
    return df


# TF-IDF
def tf_idf_sklearn(df, drop_one_letter=False):
    if drop_one_letter:
        df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [word for word in x if len(word) > 1])
    df['joined tokens'] = df['text_tokenized'].apply(
        lambda x: str.join(' ', x).replace(
            '오븐 엔조이', '오븐엔조이').replace(
            '휘 낭시', '휘낭시에').replace(
            '베이 킹', '베이킹').replace(
            '스 메그', '스메그'))
    df['text_tokenized'] = df['joined tokens'].apply(lambda x: x.split(' '))

    vectorizer = TfidfVectorizer(min_df=0.01)
    sparse_matrix = vectorizer.fit_transform(df['joined tokens'])
    return vectorizer, sparse_matrix, df


def count_sklearn(df, drop_one_letter=False):
    if drop_one_letter:
        df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [word for word in x if len(word) > 1])
    df['joined tokens'] = df['text_tokenized'].apply(
        lambda x: str.join(' ', x).replace(
            '오븐 엔조이', '오븐엔조이').replace(
            '휘 낭시', '휘낭시에').replace(
            '베이 킹', '베이킹').replace(
            '스 메그', '스메그'))
    df['text_tokenized'] = df['joined tokens'].apply(lambda x: x.split(' '))

    vectorizer = CountVectorizer(min_df=0.01)
    sparse_matrix = vectorizer.fit_transform(df['joined tokens'])
    return vectorizer, sparse_matrix


def drop_certain_words(corpus, sparse_matrix, drop_words):
    drop_words_index = [np.where(corpus == word)[0][0] for word in drop_words]
    to_keep = sorted(set(range(sparse_matrix.shape[1])) - set(drop_words_index))
    corpus = corpus[to_keep]
    sparse_matrix = sparse_matrix[:, to_keep]
    return corpus, sparse_matrix


def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        important_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

        print("Topic %d:" % topic_idx)
        print(" ".join(important_words))
        topics.append(important_words)
    return topics


def doc_labeling(df, matrix, corpus, model):
    topics = display_topics(model, corpus, 10)
    topics_df = pd.DataFrame(topics)

    topic_dist = model.transform(matrix)
    df['topic label'] = topic_dist.argmax(1)
    df['topic prob'] = topic_dist.max(1)
    return df, topics_df


def analysis(df, n_topics):
    vectorizer, matrix, df = tf_idf_sklearn(df)

    corpus = np.array(vectorizer.get_feature_names())
    # drop_words = ['ㅋㅋ', 'ㅋㅋㅋ', 'ㅎㅎ', 'ㅜㅜ', 'ㅠㅜ', 'ㅠㅠ', 'ㅠㅠㅠ', 'ㅠㅠㅠㅠ']
    # corpus, matrix = drop_certain_words(corpus, matrix, drop_words)
    # 원래는 사용하지 않는 단어들을 제거하는 단계가 있으나 제출용으로 샘플을 줄이면서 해당 단계가 필요없게 됨

    lda_sk = LatentDirichletAllocation(n_components=n_topics)
    lda_sk.fit(matrix)

    df, topics_df = doc_labeling(df, matrix, corpus, lda_sk)
    return df, topics_df, lda_sk


# Pass optimization
def pass_opt(corpus, dictionary, data, min_pass, max_pass):
    perplexity_value = []
    coherence_value = []

    for p in range(min_pass, max_pass):
        print(p)
        ldamodel = gensim.models.LdaMulticore(corpus, num_topics=20, id2word=dictionary, passes=p, workers=16)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=data['text_tokenized'],
                                             dictionary=dictionary, topn=10, coherence='c_v')
        perplexity_value.append(ldamodel.log_perplexity(corpus))
        coherence_value.append(coherence_model_lda.get_coherence())

    # graphing_opt_res(coherence_value, perplexity_value, min_pass, max_pass)

    perplexity = pd.DataFrame(perplexity_value, index=range(min_pass, max_pass), columns=['perplexity'])
    coherence = pd.DataFrame(coherence_value, index=range(min_pass, max_pass), columns=['coherence'])
    parameter_tuning = pd.concat([perplexity, coherence], axis=1)

    return parameter_tuning


# Num_topic optimization
def lda_param_opt(corpus, dictionary, data, pass_value=20, min_topic=2, max_topic=40):
    perplexity_value = []
    coherence_value = []
    for i in range(min_topic, max_topic + 1):
        print(i)
        ldamodel = gensim.models.LdaMulticore(corpus, num_topics=i, id2word=dictionary,
                                              passes=pass_value, workers=16)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=data['text_tokenized'],
                                             dictionary=dictionary, topn=10, coherence='c_v')
        perplexity_value.append(ldamodel.log_perplexity(corpus))
        coherence_value.append(coherence_model_lda.get_coherence())

    # graphing_opt_res(coherence_value, perplexity_value, min_topic, max_topic)

    perplexity = pd.DataFrame(perplexity_value, index=range(min_topic, max_topic+1), columns=['perplexity'])
    coherence = pd.DataFrame(coherence_value, index=range(min_topic, max_topic+1), columns=['coherence'])
    parameter_tuning = pd.concat([perplexity, coherence], axis=1)

    return parameter_tuning


# LDA
def time_series_analysis(df):
    df['날짜'] = pd.to_datetime(df['날짜'])
    period_df = df.copy()
    period_df.index = period_df['날짜'].values
    period_df['날짜'] = [date.date() for date in period_df.index]
    period_df['count'] = np.ones(len(period_df))

    whole_period = period_df[['topic label', 'count']].groupby('topic label').sum().sort_values('count',
                                                                                                ascending=False)
    whole_period_percentage = whole_period / whole_period.sum()

    time_series = period_df[['날짜', 'topic label', 'count']].groupby(['날짜', 'topic label']).sum().reset_index()
    time_series_result = time_series.pivot_table(columns=['topic label'], index=['날짜'], values='count').fillna(0)
    time_series_result.index = pd.to_datetime(time_series_result.index)
    monthly_result = time_series_result.loc['2018-06-01':].resample('3M').sum()
    monthly_percentage = (monthly_result.T / monthly_result.sum(1)).T

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_percentage)
    plt.legend(['Topic %d' % (number+1) for number in monthly_percentage.columns])
    plt.show()

    return whole_period_percentage, monthly_percentage, time_series_result


def closer_look(df, topic_num, content, limit=40):
    each_topic_df = df[df['topic label'] == topic_num]
    print(each_topic_df[['topic prob', '제목', '본문', '댓글']].sort_values(by='topic prob', ascending=False)[content][:limit])
