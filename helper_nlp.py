import os
import re
import json

import pandas as pd
import matplotlib.pyplot as plt

import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

from konlpy.tag import Okt

os.environ['JAVA_HOME'] = '/usr/bin/java'


def data_loading():
    raw_data1 = pd.read_json('data_baking/baking_v2.json', orient='table').dropna()
    raw_data2 = pd.read_json('data_baking/baking_v2_cont.json', orient='table').dropna()
    raw_data1.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...
    raw_data2.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...

    raw_data3 = pd.read_json('data_baking_equip/crawled_texts_baking_equip.json', orient='table').dropna()
    raw_data4 = pd.read_json('data_baking_equip/equip_v2_cont.json', orient='table').dropna()
    raw_data4.columns = ['날짜', '조회수', '댓글개수', '좋아요', '제목', '닉네임', '본문', '댓글']  # 크롤링 과정에서 칼럽 라벨링 잘못됨...

    baking_data = pd.concat([raw_data1, raw_data2], ignore_index=True)
    equip_data = pd.concat([raw_data3, raw_data4], ignore_index=True)

    return baking_data, equip_data


def parsed_data_loading():
    baking_data = pd.read_json('parsed_data/parsed_baking.json', orient='table')
    equip_data = pd.read_json('parsed_data/parsed_equip.json', orient='table')
    return baking_data, equip_data


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
    df['clean title'] = df['제목'].apply(clean_text)
    df['clean text'] = df['본문'].apply(clean_text)
    return df


# 형태소 분석
def tokenization(df, nouns=True):
    okt = Okt()

    if nouns:
        target_title = [okt.nouns(doc) for doc in df['clean title']]
        target_text = [okt.nouns(doc) for doc in df['clean text']]

    else:
        target_title = [okt.morphs(doc) for doc in df['clean title']]
        target_text = [okt.morphs(doc) for doc in df['clean text']]

    # 불용어 제거
    with open('nlp_data/korean_stopwords.json', encoding='utf-8') as f:
        stopwords = json.load(f)
    stopwords.extend(['에서', '고', '이다', '는', '이', '가', '한', '씨', '"', '에', '요', '걸', '더', '케', '거', '분'])

    df['title_tokenized'] = target_title
    df['title_tokenized'] = df['title_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])

    df['text_tokenized'] = target_text
    df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [a for a in x if a not in stopwords])
    return df


# TF-IDF
def tf_idf(df, drop_one_letter=False):
    if drop_one_letter:
        df['text_tokenized'] = df['text_tokenized'].apply(lambda x: [word for word in x if len(word) > 1])
    df['joined tokens'] = df['text_tokenized'].apply(
        lambda x: str.join(' ', x).replace(
            '오븐 엔조이', '오븐엔조이').replace(
            '휘 낭시', '휘낭시에'). replace(
            '베이 킹', '베이킹').replace(
            '스 메그', '스메그'))
    df['text_tokenized'] = df['joined tokens'].apply(lambda x: x.split(' '))

    dictionary = gensim.corpora.Dictionary(df['text_tokenized'])
    bow_corpus = [dictionary.doc2bow(doc) for doc in df['text_tokenized']]
    tfidf = gensim.models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    return tfidf_corpus, dictionary, df


# LDA
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

    graphing_opt_res(coherence_Value, perplexity_value, min_topics, max_topic)

    perplexity = pd.DataFrame(perplexity_value, index=range(min_value, max_value), columns=['perplexity'])
    coherence = pd.DataFrame(coherence_value, index=range(min_value, max_value), columns=['coherence'])
    parameter_tuning = pd.concat([perplexity, coherence], axis=1)

    return parameter_tuning


# Graphing
def graphing_opt_res(coherence_value, perplexity_value, min_value, max_value):
    x = range(min_value, max_value)
    plt.figure()
    plt.plot(x, perplexity_value)
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.show()

    plt.figure()
    plt.plot(x, coherence_value)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence')
    plt.show()
